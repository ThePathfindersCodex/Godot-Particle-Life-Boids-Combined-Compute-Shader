#[compute]
#version 450
layout(local_size_x = 512, local_size_y = 1, local_size_z = 1) in;
shared uint prefix_sum_temp[512]; // adjust size to local_size_x

struct MyVec2 { vec2 v; };

// === Input Buffers ===
layout(set = 0, binding = 0, std430) buffer InPosBuffer {
    MyVec2 data[];
} in_pos_buffer;

layout(set = 0, binding = 1, std430) buffer InVelBuffer {
    MyVec2 data[];
} in_vel_buffer;

// Oer Agent Species/type index
layout(set = 0, binding = 2, std430) buffer InSpeciesBuffer {
    int data[];
} in_species_buffer;

// === Output Buffers ===
layout(set = 0, binding = 3, std430) buffer OutPosBuffer {
    MyVec2 data[];
} out_pos_buffer;

layout(set = 0, binding = 4, std430) buffer OutVelBuffer {
    MyVec2 data[];
} out_vel_buffer;

// === Species Interaction Matrix ===
// (Flattened as species_count x species_count float array)
layout(set = 0, binding = 5, std430) readonly buffer MatrixBuffer {
    float data[];
} interaction_matrix;

// === Collision Buffers ===
// Per-agent collision counts
layout(set = 0, binding = 6, std430) buffer CollisionCountBuffer {
    uint count[];
} collision_count_buffer;

// Per-agent fixed-size collision partner list
layout(set = 0, binding = 7, std430) buffer CollisionPartnerBuffer {
    uint partners[];
} collision_partner_buffer;

// === Spatial Hashing Buffers ===
// spatial hashing: per-cell counts
layout(set = 0, binding = 8, std430) buffer CellCountBuffer {
    uint cell_counts[]; // length = num_cells
} cell_count_buffer;

// spatial hashing: per-cell offsets
layout(set = 0, binding = 9, std430) buffer CellOffsetBuffer {
    uint cell_offsets[]; // length = num_cells
} cell_offset_buffer;

// spatial hashing: sorted indices: list of agent ids grouped by cell
layout(set = 0, binding = 10, std430) buffer SortedIndexBuffer {
    uint sorted_indices[]; // length = agents_count
} sorted_index_buffer;

// spatial hashing: agent cells
layout(set = 0, binding = 11, std430) buffer AgentCellBuffer {
    uint data[];  // length = agents_count
} agent_cell_buffer;

// spatial hashing: cursor
layout(set = 0, binding = 12, std430) buffer CursorBuffer {
    uint data[];  // length = num_cells
} cursor_buffer;

// Render target
layout(set = 0, binding = 13, rgba32f) uniform image2D OUTPUT_TEXTURE;

// === Parameters ===
layout(push_constant, std430) uniform Params {

    float run_mode;             // 0 = sim, 1 = collide, 2 = clear, 3 = draw
    float dt;                   // Timestep
	float mix_t;         		// Mix Boids with PLife
    float agents_count;         // Total agent count
    float species_count;        // Number of species

	float boid_vision_radius;   // Neighbor radius for flocking
    float species_interaction_radius; // Range of inter-species forces
	
	float alignment_force;      // Align with nearby velocities
    float cohesion_force;       // Pull toward neighbor center
    float separation_force;     // Push away when too close
	
    float movement_randomness;  // Random motion component
    float movement_scaling;     // Global motion multiplier	
    float force_softening;            // Avoids infinite forces at zero dist
    float center_attraction;          // Pull toward scene center
    float drag;                 // Velocity damping
    float min_speed;            // Clamp lower speed
    float max_speed;            // Clamp upper speed
    float max_force;            // Limit total applied force

    float collision_radius;           // Physical collision distance
	float max_collisions;         // how many collides to resolve

    float cell_size;        // hashing cell size
    float cells_per_row;    // hashing cells per row

    // --- Rendering & camera ---
    float draw_radius;          // Agent display size
    float image_size;           // Render target dimension
    float world_size_mult;       // Scales worlds for flocking	
    float camera_center_x;      // View pan X
    float camera_center_y;      // View pan Y
    float zoom;                 // Camera zoom
} params;

//////////////////////

// Clamp vector magnitude
vec2 limit(vec2 v, float max_val) {
    float mag = length(v);
    if (mag > max_val)
        return normalize(v) * max_val;
    return v;
}

// Safe normalize to avoid NaN on zero-length vectors
vec2 safe_normalize(vec2 v) {
    float len = length(v);
    return (len > 0.0001) ? v / len : vec2(0.0);
}

// Random direction based on integer ID
vec2 random_dir(uint id, float scale) {
    uint seed = id * 1664525u + 1013904223u;
    float ang = float(seed % 6283u) * 0.001f;
    return vec2(cos(ang), sin(ang)) * scale;
}

// Toroidal distance difference (wrap-around world)
vec2 toroidal_diff(vec2 a, vec2 b, vec2 world_size) {
    vec2 d = b - a;
    d -= world_size * round(d / world_size);
    return d;
}

// Apply wrapping border
void apply_border(inout vec2 pos, inout vec2 vel) {
    float world_size = params.image_size * params.world_size_mult;
    float half_size = world_size * 0.5;
    if (pos.x < -half_size) pos.x += world_size;
    if (pos.x >  half_size) pos.x -= world_size;
    if (pos.y < -half_size) pos.y += world_size;
    if (pos.y >  half_size) pos.y -= world_size;
}

// Apply softened and capped force
float apply_force(float f, float dist, float softening, float max_force) {
    float softened_dist = sqrt(dist * dist + softening * softening);
    float force_mag = f / softened_dist;
    return clamp(force_mag, -max_force, max_force);
}

void run_sim() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= uint(params.agents_count)) return;

    vec2 pos = in_pos_buffer.data[id].v;
    vec2 vel = in_vel_buffer.data[id].v;
    int species = in_species_buffer.data[id];

    // World and accumulators
    float world_size_d = params.image_size * params.world_size_mult;
    vec2 world_size = vec2(world_size_d);
    vec2 align = vec2(0.0);
    vec2 coh   = vec2(0.0);
    vec2 sep   = vec2(0.0);
    vec2 interact = vec2(0.0);
    vec2 coll = vec2(0.0);
    int neighbor_count = 0;

    // Compute world-to-grid conversion
    float world_size_f = params.image_size * params.world_size_mult;
    float half_world = 0.5 * world_size_f;
    float cs = params.cell_size;
    int cpr = int(params.cells_per_row);

    // compute this agent's cell coords (wrap/toroidal)
    vec2 pos_wrapped = pos + vec2(half_world);
    pos_wrapped = mod(pos_wrapped + world_size_f, world_size_f);
    int cx = int(floor(pos_wrapped.x / cs)) % cpr;
    int cy = int(floor(pos_wrapped.y / cs)) % cpr;

    // iterate over neighbor cells (3x3)
    for (int dx = -1; dx <= 1; ++dx) {
        for (int dy = -1; dy <= 1; ++dy) {
            int ncx = (cx + dx) % cpr;
            int ncy = (cy + dy) % cpr;
            if (ncx < 0) ncx += cpr;
            if (ncy < 0) ncy += cpr;
            uint cell_index = uint(ncy * cpr + ncx);

            uint start = cell_offset_buffer.cell_offsets[cell_index];
            uint count = cell_count_buffer.cell_counts[cell_index];
            uint end = start + count;

            for (uint k = start; k < end; ++k) {
                uint i = sorted_index_buffer.sorted_indices[k];
                if (i == id) continue;

                vec2 other_pos = in_pos_buffer.data[i].v;
                vec2 diff = toroidal_diff(pos, other_pos, vec2(world_size_f));
                float dist = length(diff);
                //if (dist < 0.0001) continue;
                if (dist < 0.0001) dist = 0.0001;

                // boid behavior
                if (dist < params.boid_vision_radius) {
                    neighbor_count++;
                    align += in_vel_buffer.data[i].v;
                    coh   += pos + diff;
                    sep  -= diff / (dist * dist);
                }

                // species interactions
                if (dist < params.species_interaction_radius) {
                    int other_species = in_species_buffer.data[i];
                    float f = interaction_matrix.data[
                        species * uint(params.species_count) + other_species
                    ];
                    vec2 dir = diff / dist;
                    interact += dir * apply_force(f, dist, params.force_softening, params.max_force);
                }

                // Collision recorded into collision buffers
                if (dist < params.collision_radius) {
                    uint slot = atomicAdd(collision_count_buffer.count[id], 1u);
                    uint max_collisions = uint(params.max_collisions);
                    if (slot < max_collisions) {
                        collision_partner_buffer.partners[id * max_collisions + slot] = i;
                    }
                }
            }
        }
    }

   // === Finalize BOIDS averages ===
    vec2 boid_force = vec2(0.0);
    if (neighbor_count > 0) {
        align = safe_normalize(align / neighbor_count) * params.alignment_force;
        coh   = safe_normalize((coh / neighbor_count) - pos) * params.cohesion_force;
        sep   = safe_normalize(sep) * params.separation_force;
        boid_force = align + coh + sep;
    }

    // === Combine BOIDS + PLife ===
    vec2 f_boid = boid_force;
    vec2 f_plife = interact;
    vec2 accel = mix(f_boid, f_plife, params.mix_t);
	
    // Center pull
    if (params.center_attraction > 0.0001) {
        vec2 dir_to_center = safe_normalize(-pos);
        accel += dir_to_center * params.center_attraction;
    }

    // Add small random drift
    accel += random_dir(id + uint(gl_WorkGroupID.x), params.movement_randomness);
    accel = limit(accel, params.max_force);
	
	// Global accel scaling
	accel *= params.movement_scaling;

    // Clamp speed
	float speed = length(vel);
    if (speed < params.min_speed && speed > 0.0001)
        vel = normalize(vel) * params.min_speed;
    if (speed > params.max_speed)
        vel = normalize(vel) * params.max_speed;

	// Integrate
	vel += accel * params.dt;
	vel *= params.drag; // optional damping

    pos += vel * params.dt;
    apply_border(pos, vel);

    // === Output ===
    out_pos_buffer.data[id].v = pos;
    out_vel_buffer.data[id].v = vel;
}


//////////////////////
void resolve_collide() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= uint(params.agents_count)) return;

    vec2 pos = out_pos_buffer.data[id].v;
    vec2 vel = out_vel_buffer.data[id].v;

    vec2 correction = vec2(0.0);
    uint contrib_count = 0u;

    uint max_collisions = uint(params.max_collisions);
    uint raw_count = collision_count_buffer.count[id];
    uint c = min(raw_count, max_collisions);

    if (c > uint(params.agents_count)) c = uint(params.agents_count);

    float world_size_f = params.image_size * params.world_size_mult;
    float col_radius = params.collision_radius;
    float per_neighbor_max = col_radius * 1.0; //0.5;
    float max_move = col_radius * 1.0; //0.9;
    float apply_frac = 1.0; // 0.5;
	// float damping = 1.0;
    // float max_vel_change = length(vec2(col_radius, col_radius));

    for (uint s = 0u; s < c; ++s) {
        uint j = collision_partner_buffer.partners[id * max_collisions + s];
        if (j >= uint(params.agents_count) || j == id) continue;

        vec2 other_pos = out_pos_buffer.data[j].v;
        // vec2 other_vel = out_vel_buffer.data[j].v;

        vec2 diff = toroidal_diff(pos, other_pos, vec2(world_size_f));
        diff = -diff;

        float dist = length(diff);
        if (dist < 1e-6) {
            float angle = float((id + 37u) % 1024u) * 0.0062831853;
            vec2 n = vec2(cos(angle), sin(angle));
            float overlap = col_radius;
            float single_contrib = min(overlap, per_neighbor_max);
            correction += n * single_contrib;
            contrib_count++;
            continue;
        }

        if (dist < col_radius) {
            vec2 n = diff / dist;
            float overlap = col_radius - dist;
            float single_contrib = min(overlap, per_neighbor_max);
            correction += n * single_contrib;
            contrib_count++;

            // float rel_vn = dot(vel - other_vel, n);
            // if (rel_vn < 0.0) {
                
                // float dv = -rel_vn * damping;
                // dv = min(dv, max_vel_change);
                // vel -= n * dv;
            // }
        }
    }

    if (contrib_count > 0u) {
        correction /= float(contrib_count);
        correction = clamp(correction, -vec2(max_move), vec2(max_move));
        pos += correction * apply_frac;
    }

    out_pos_buffer.data[id].v = pos;
    // out_vel_buffer.data[id].v = vel;
}
//////////////////////

vec3 heatmap_color(float t) {
    // Clamp between 0 and 1
    t = clamp(t, 0.0, 1.0);

    // Map t to blue to cyan to green to yellow to red
    if (t < 0.25) {
        // Blue to Cyan
        float k = t / 0.25;
        return mix(vec3(0.0, 0.0, 1.0), vec3(0.0, 1.0, 1.0), k);
    } else if (t < 0.5) {
        // Cyan to Green
        float k = (t - 0.25) / 0.25;
        return mix(vec3(0.0, 1.0, 1.0), vec3(0.0, 1.0, 0.0), k);
    } else if (t < 0.75) {
        // Green to Yellow
        float k = (t - 0.5) / 0.25;
        return mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), k);
    } else {
        // Yellow to Red
        float k = (t - 0.75) / 0.25;
        return mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), k);
    }
}

vec3 species_color_dynamic(int species) {
    // Map species index to 0..1 range
    float t = float(species) / max(float(params.species_count - 1), 1.0);
    return heatmap_color(t);
}

void draw_circle(vec2 center, float radius, vec4 color) {
    ivec2 min_pix = ivec2(floor(center - radius));
    ivec2 max_pix = ivec2(ceil(center + radius));
	float r2 = radius * radius;  // squared radius
    for (int x = min_pix.x; x <= max_pix.x; x++) {
        for (int y = min_pix.y; y <= max_pix.y; y++) {
            vec2 diff = vec2(x, y) - center;
            if (dot(diff, diff) <= r2) {  // squared distance check
                imageStore(OUTPUT_TEXTURE, ivec2(x, y), color);
            }
        }
    }
}

void clear_texture() {
    ivec2 pixel = ivec2(gl_GlobalInvocationID.xy);
    //imageStore(OUTPUT_TEXTURE, pixel, vec4(vec3(0.1),1.0)); // slight off-black	
    imageStore(OUTPUT_TEXTURE, pixel, vec4(vec3(1.0),1.0)); // white
}

void draw_texture() {
    uint id = gl_GlobalInvocationID.x;
    if (id >= uint(params.agents_count)) return;

    vec2 curr_pos = out_pos_buffer.data[id].v;   // Current particle position
    vec2 image_size_vec = vec2(params.image_size, params.image_size);
    int species = in_species_buffer.data[id];
	float draw_size = params.draw_radius * params.zoom;

    vec2 rel = curr_pos - vec2(params.camera_center_x, params.camera_center_y);
    rel *= params.zoom;
    vec2 screen_pos = rel + image_size_vec * 0.5;

    if (screen_pos.x < -draw_size || screen_pos.x >= image_size_vec.x + draw_size ||
		screen_pos.y < -draw_size || screen_pos.y >= image_size_vec.y + draw_size) {
		return;
    }

    vec3 color = species_color_dynamic(species);
    //vec3 color = vec3(0.0, 1.0, 0.0);

    // Draw a circle for the particle
	draw_size = max(draw_size, 1.0);
    draw_circle(screen_pos, draw_size, vec4(color, 1.0));
}

// ------------------ ZERO OUT COUNT AND COLLIDE BUFFERS ------------------
void zero_counts() {
    uint cid = gl_GlobalInvocationID.x;
    if (cid >= uint(params.cells_per_row * params.cells_per_row)) return;
    cell_count_buffer.cell_counts[cid] = 0u;

    for (uint i = 0u; i < params.max_collisions; ++i)
		collision_partner_buffer.partners[cid * uint(params.max_collisions) + i] = 0u;
}
// ------------------ COUNT CELLS ------------------
void count_cells() {
    uint id = gl_GlobalInvocationID.x; // compute directly
    if (id >= uint(params.agents_count)) return;

    vec2 p = in_pos_buffer.data[id].v;  // use in_pos_buffer
    float half_size = params.image_size * params.world_size_mult * 0.5; // matches previous world_size calc
    float rx = mod(p.x + half_size + params.image_size * params.world_size_mult, params.image_size * params.world_size_mult);
    float ry = mod(p.y + half_size + params.image_size * params.world_size_mult, params.image_size * params.world_size_mult);
    
    int cx = int(floor(rx / params.cell_size)) % int(params.cells_per_row);
    int cy = int(floor(ry / params.cell_size)) % int(params.cells_per_row);
    uint cell = uint(cy * int(params.cells_per_row) + cx);

    agent_cell_buffer.data[id] = cell; // buffer for agent -> cell mapping

    atomicAdd(cell_count_buffer.cell_counts[cell], 1u); // increment per-cell count
}

void prefix_sum() {
    uint tid = gl_LocalInvocationID.x;
    uint gid = gl_GlobalInvocationID.x;
    uint num_cells = uint(params.cells_per_row * params.cells_per_row);
    if (gid >= num_cells) return;

    prefix_sum_temp[tid] = cell_count_buffer.cell_counts[gid]; // use cell_count_buffer
    barrier();

    // Hillis-Steele inclusive scan
    for (uint offset = 1u; offset < gl_WorkGroupSize.x; offset <<= 1u) {
        uint n = 0u;
        if (tid >= offset) n = prefix_sum_temp[tid - offset];
        barrier();
        prefix_sum_temp[tid] += n;
        barrier();
    }

    // exclusive scan
    uint excl = (tid == 0u) ? 0u : prefix_sum_temp[tid - 1u];
    cell_offset_buffer.cell_offsets[gid] = excl; // write to offsets buffer
}
// ------------------ COPY OFFSETS TO CURSOR ------------------
void copy_offsets_to_cursor() {
    uint cid = gl_GlobalInvocationID.x;
    uint num_cells = uint(params.cells_per_row * params.cells_per_row);
    if (cid >= num_cells) return;

    cursor_buffer.data[cid] = cell_offset_buffer.cell_offsets[cid]; // use cursor buffer
}
// ------------------ SCATTER SORTED INDICES ------------------
void scatter_sorted_indices() {
    uint id = gl_GlobalInvocationID.x; // compute directly
	if (id >= uint(params.agents_count)) return;

    uint cell = agent_cell_buffer.data[id]; // use agent_cell_buffer
    uint pos = atomicAdd(cursor_buffer.data[cell], 1u); // use cursor_buffer
    sorted_index_buffer.sorted_indices[pos] = id;        // write to final sorted indices
}

void main() {
    if (params.run_mode == 0 && params.dt > 0.0) {
        run_sim();
    } else if (params.run_mode == 1 && params.dt > 0.0) {
        resolve_collide();
    } else if (params.run_mode == 2) {
        clear_texture();
    } else if (params.run_mode == 3) {
        draw_texture();
    }
	
	// ---- new GPU-only preprocessing modes ----
    else if (params.run_mode == 10) {    // ZERO_COUNTS
        zero_counts();
    } else if (params.run_mode == 11) {  // COUNT_CELLS
        count_cells();
    } else if (params.run_mode == 12) {  // PREFIX_SUM
        prefix_sum();
    } else if (params.run_mode == 13) {  // COPY_OFFSETS
        copy_offsets_to_cursor();
    } else if (params.run_mode == 14) {  // SCATTER
        scatter_sorted_indices();
    }	
}
