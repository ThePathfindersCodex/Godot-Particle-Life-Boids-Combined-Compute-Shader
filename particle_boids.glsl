#[compute]
#version 450
layout(local_size_x = 16, local_size_y = 16, local_size_z = 1) in;
shared uint prefix_sum_temp[256]; // adjust size to local_size_x*local_size_y
shared uint block_base;

struct MyVec2 { vec2 v; };

// === Input/Output Buffers ===
layout(rgba32f, set = 0, binding = 0) uniform restrict image2D input_particles; // R=pos.x, G=pos.y, B=vel.x, A=vel.y
layout(rgba32f, set = 1, binding = 0) uniform restrict image2D output_particles;

// Per Agent Species/type index
layout(set = 0, binding = 1, std430) buffer InSpeciesBuffer { int data[]; }   in_species_buffer;

// === Species Interaction Matrix ===
// (Flattened as species_count x species_count float array)
layout(set = 0, binding = 2, std430) readonly buffer MatrixBuffer {
    float data[];
} interaction_matrix;

// === Collision Buffers ===
// Per-agent collision counts
layout(set = 0, binding = 3, std430) buffer CollisionCountBuffer {
    uint count[];
} collision_count_buffer;

// Per-agent fixed-size collision partner list
layout(set = 0, binding = 4, std430) buffer CollisionPartnerBuffer {
    uint partners[];
} collision_partner_buffer;

// === Spatial Hashing Buffers ===
// spatial hashing: per-cell counts
layout(set = 0, binding = 5, std430) buffer CellCountBuffer {
    uint cell_counts[]; // length = num_cells
} cell_count_buffer;

// spatial hashing: per-cell offsets
layout(set = 0, binding = 6, std430) buffer CellOffsetBuffer {
    uint cell_offsets[]; // length = num_cells
} cell_offset_buffer;

// spatial hashing: sorted indices: list of agent ids grouped by cell
layout(set = 0, binding = 7, std430) buffer SortedIndexBuffer {
    uint sorted_indices[]; // length = agents_count
} sorted_index_buffer;

// spatial hashing: agent cells
layout(set = 0, binding = 8, std430) buffer AgentCellBuffer {
    uint data[];  // length = agents_count
} agent_cell_buffer;

// spatial hashing: cursor
layout(set = 0, binding = 9, std430) buffer CursorBuffer {
    uint data[];  // length = num_cells
} cursor_buffer;

// === Parameters ===
layout(push_constant, std430) uniform Params {

    float run_mode;             // determine which logic to run on GPU
    float dt;                   // Timestep
	float compute_texture_size; // Size of the data texture

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

    float image_size;           // Render target dimension
    float world_size_mult;       // Scales worlds for flocking	
} params;

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

// Main simulation logic (using spatial grid) + prepare collisions.
void run_sim() {
	ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
	int id = int(uv.y * params.compute_texture_size + uv.x);

	if (id >= params.agents_count || uv.x >= params.compute_texture_size || uv.y >= params.compute_texture_size) {
		return;
	}
	vec4 pixel = imageLoad(input_particles, uv);

    vec2 pos = pixel.rg;
    vec2 vel = pixel.ba;
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
            uint end   = start + cell_count_buffer.cell_counts[cell_index];
			
            for (uint k = start; k < end; ++k) {
                uint other = sorted_index_buffer.sorted_indices[k];
                if (other == id) continue;
				
				ivec2 other_uv = ivec2(other % int(params.compute_texture_size), other / params.compute_texture_size);
				vec4 other_pixel = imageLoad(input_particles, other_uv);
				
                vec2 other_pos = other_pixel.rg;
				vec2 other_vel = other_pixel.ba;
				int other_species = in_species_buffer.data[other];

                vec2 diff = toroidal_diff(pos, other_pos, vec2(world_size_f));
                float dist = length(diff);
                //if (dist < 0.0001) continue;
                if (dist < 0.0001) dist = 0.0001;

                // boid behavior
                if (dist < params.boid_vision_radius) {
                    neighbor_count++;
                    //align += in_vel_buffer.data[i].v;
					align += other_vel;
                    coh   += pos + diff;
                    sep  -= diff / (dist * dist);
                }

                // species interactions
                if (dist < params.species_interaction_radius) {
                    //int other_species = in_species_buffer.data[i];
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
                        collision_partner_buffer.partners[id * max_collisions + slot] = other;
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
    accel += random_dir(id, params.movement_randomness);
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
	imageStore(output_particles, uv, vec4(pos, vel));
}


void resolve_collide() {
	ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
	int id = int(uv.y * params.compute_texture_size + uv.x);
	
	if (id >= params.agents_count || uv.x >= params.compute_texture_size || uv.y >= params.compute_texture_size) {
		return;
	}
	//vec4 pixel = imageLoad(input_particles, uv);
	vec4 pixel = imageLoad(output_particles, uv);

    vec2 pos = pixel.rg;
    vec2 vel = pixel.ba;
    int species = in_species_buffer.data[id];

    vec2 correction = vec2(0.0);
    uint contrib_count = 0u;

    uint max_collisions = uint(params.max_collisions);
    uint raw_count = collision_count_buffer.count[id];
    uint c = min(raw_count, max_collisions);

    if (c > uint(params.agents_count)) c = uint(params.agents_count);

    float world_size_f = params.image_size * params.world_size_mult;
    float col_radius = params.collision_radius;
    float per_neighbor_max = col_radius * 2.0; // 0.5;
    float max_move = col_radius * 1.0; // 0.9;
    float apply_frac = 1.0; // 0.5;

    for (uint s = 0u; s < c; ++s) {
        uint j = collision_partner_buffer.partners[id * max_collisions + s];
        if (j >= uint(params.agents_count) || j == id) continue;

		uint other = j;
		if (other == id) continue;
		
		ivec2 other_uv = ivec2(other % int(params.compute_texture_size), other / params.compute_texture_size);
		//vec4 other_pixel = imageLoad(input_particles, other_uv);
		vec4 other_pixel = imageLoad(output_particles, other_uv);
		
		vec2 other_pos = other_pixel.rg;
		//vec2 other_vel = other_pixel.ba;

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
        }
    }

    if (contrib_count > 0u) {
        correction /= float(contrib_count);
        correction = clamp(correction, -vec2(max_move), vec2(max_move));
        pos += correction * apply_frac;
    }

	//imageStore(output_particles, uv, vec4(pos, vel));
	imageStore(input_particles, uv, vec4(pos, vel));
}

void count_cells() {
	ivec2 uv = ivec2(gl_GlobalInvocationID.xy);
	int id = int(uv.y * params.compute_texture_size + uv.x);
	
	if (id >= params.agents_count || uv.x >= params.compute_texture_size || uv.y >= params.compute_texture_size) {
		return;
	}

	vec4 pixel = imageLoad(input_particles, uv);
	
	vec2 p = pixel.rg;
	
    float half_size = params.image_size * params.world_size_mult * 0.5; // matches world_size calc
    float rx = mod(p.x + half_size + params.image_size * params.world_size_mult, params.image_size * params.world_size_mult);
    float ry = mod(p.y + half_size + params.image_size * params.world_size_mult, params.image_size * params.world_size_mult);
    
    int cx = int(floor(rx / params.cell_size)) % int(params.cells_per_row);
    int cy = int(floor(ry / params.cell_size)) % int(params.cells_per_row);
    uint cell = uint(cy * int(params.cells_per_row) + cx);

    agent_cell_buffer.data[id] = cell; // buffer for agent -> cell mapping

    atomicAdd(cell_count_buffer.cell_counts[cell], 1u); // increment per-cell count
}

void prefix_sum() {
    const uint L = gl_WorkGroupSize.x * gl_WorkGroupSize.y; // 256u; // adjust size to local_size_x*local_size_y
    uint tid =
        gl_LocalInvocationID.y * gl_WorkGroupSize.x +
        gl_LocalInvocationID.x;
    uint group_id = gl_WorkGroupID.x;
    uint num_cells = uint(params.cells_per_row) * uint(params.cells_per_row);

    if (group_id == 0u && tid == 0u) cursor_buffer.data[0] = 0u;
    barrier();

    uint val = 0u;
    uint index = group_id * L + tid;
    if (index < num_cells) val = cell_count_buffer.cell_counts[index];
    prefix_sum_temp[tid] = val;
    barrier();

    for (uint offset = 1u; offset < L; offset <<= 1u) {
        uint step = offset << 1u;
        uint ix = (tid + 1u) * step - 1u;
        if (ix < L) prefix_sum_temp[ix] += prefix_sum_temp[ix - offset];
        barrier();
    }

    uint block_total = prefix_sum_temp[L - 1u];
    if (tid == 0u) prefix_sum_temp[L - 1u] = 0u;
    barrier();

    for (uint offset = L >> 1u; offset >= 1u; offset >>= 1u) {
        uint step = offset << 1u;
        uint ix = (tid + 1u) * step - 1u;
        if (ix < L) {
            uint t = prefix_sum_temp[ix - offset];
            prefix_sum_temp[ix - offset] = prefix_sum_temp[ix];
            prefix_sum_temp[ix] += t;
        }
        barrier();
        if (offset == 1u) break;
    }

	if (tid == 0u) block_base = atomicAdd(cursor_buffer.data[0], block_total);
	barrier();
	uint base = block_base;

    if (index < num_cells) {
        uint offset_for_cell = base + prefix_sum_temp[tid];
        cell_offset_buffer.cell_offsets[index] = offset_for_cell;
        cursor_buffer.data[index] = offset_for_cell;
    }
}

void scatter_sorted_indices() {
	uint width = gl_NumWorkGroups.x * gl_WorkGroupSize.x;
    uint id = gl_GlobalInvocationID.y * width + gl_GlobalInvocationID.x;
	if (id >= uint(params.agents_count)) return;

    uint cell = agent_cell_buffer.data[id]; // agent_cell_buffer
    uint pos = atomicAdd(cursor_buffer.data[cell], 1u); // cursor_buffer
    sorted_index_buffer.sorted_indices[pos] = id; // write to final sorted indices
}

void main() {
	// ---- GPU processing modes ----
    if (params.run_mode == 0 && params.dt > 0.0) {
        run_sim();
    } else if (params.run_mode == 1 && params.dt > 0.0) {
        resolve_collide();

	// ---- GPU preprocessing modes ----
    } else if (params.run_mode == 10 && params.dt > 0.0) {  // COUNT CELLS
        count_cells();
    } else if (params.run_mode == 11 && params.dt > 0.0) {  // PREFIX SUM
        prefix_sum();
    } else if (params.run_mode == 12 && params.dt > 0.0) {  // SCATTER
        scatter_sorted_indices();
    }
}