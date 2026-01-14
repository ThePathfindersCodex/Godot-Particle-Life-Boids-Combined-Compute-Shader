extends TextureRect

# CONFIG
var shader_local_size := 512
var empty_img : Image
@onready var image_size : int = %ParticleBoids.size.x:
	set(v):
		empty_img = Image.create(v, v, false, Image.FORMAT_RGBAF)
		empty_img.fill(Color(0.1, 0.1, 0.1, 1.0))
		image_size = v
var world_size_mult : int = 20
var agent_count : int = 1024*25
var species_count : int = 10
var draw_radius : float = 2.0

# STARTUP PARAMS
var starting_method : int = 4 # method to use when restarting new field?
var rand_start_interaction_range : float = 2.0 # force will be random between -X and +X
var rand_start_radius_mul : float = 16.0 # different startup patterns use this multiplier
var start_agent_count : int = agent_count # only used when restarting new field
var start_species_count : int = species_count # only used when restarting new field

# SPEED/TIME
var dt : float = .25
var paused_dt : float = dt # only used for pause/resume feature

# MIX SIMS
var mix_t: float = 0.5 # [0.0 == full boids; == 1.0 full particle life]

# VISION KERNELS
var boid_vision_radius : float = 350.0
var species_interaction_radius : float = 250.0

# BOIDS PARAMETERS
var alignment_force : float = 1.0
var cohesion_force : float = 1.0
var separation_force : float = 1.0

# FORCE ADJUSTMENTS
var movement_randomness : float = 0.01
var movement_scaling : float = 1.0
var force_softening_mul : float = 3.0
var force_softening : float = species_interaction_radius * force_softening_mul:
	get():
		return species_interaction_radius * force_softening_mul
var center_attraction : float = 0.00 # set to 0 to turn off
var damping : float = 0.98 # FRICTION
var min_speed : float = 0.0
var max_speed : float = 500.0
var max_force : float = 1000.0

# COLLISION FORCE
const MAX_COLLISIONS := 64 # 32 # 64 # 128  # tune as needed
var collision_modifier : float = 2.0
var collision_radius : float =  draw_radius + collision_modifier:
	get():
		return draw_radius + collision_modifier

# CAMERA
var camera_center : Vector2 = Vector2.ZERO
var zoom : float = 0.1
const MIN_ZOOM := 0.05
const MAX_ZOOM := 2.0

# SPATIAL HASHING
var cell_size : int = 0
var cells_per_row : int = 0
var num_cells : int = 0

# INTERACTION MATRIX
var interaction_matrix : PackedFloat32Array = []

# RENDERER SETUP
var rdmain := RenderingServer.get_rendering_device()
var textureRD: Texture2DRD
var shader : RID
var pipeline : RID
var uniform_set : RID
var output_tex : RID
var fmt := RDTextureFormat.new()
var view := RDTextureView.new()
var buffers : Array[RID] = []
var uniforms : Array[RDUniform] = []
var output_tex_uniform : RDUniform

func _ready():
	randomize()
	image_size = %ParticleBoids.size.x
	fmt.width = image_size
	fmt.height = image_size
	fmt.format = RenderingDevice.DATA_FORMAT_R32G32B32A32_SFLOAT
	fmt.usage_bits = RenderingDevice.TEXTURE_USAGE_CAN_UPDATE_BIT \
					| RenderingDevice.TEXTURE_USAGE_STORAGE_BIT \
					| RenderingDevice.TEXTURE_USAGE_CAN_COPY_FROM_BIT \
					| RenderingDevice.TEXTURE_USAGE_CPU_READ_BIT \
					| RenderingDevice.TEXTURE_USAGE_SAMPLING_BIT
	view = RDTextureView.new()
	textureRD = Texture2DRD.new()
	
	RenderingServer.call_on_render_thread(restart_simulation)

func _exit_tree():
	if textureRD:
		textureRD.texture_rd_rid = RID()
	RenderingServer.call_on_render_thread(_free_compute_resources)

func _free_compute_resources():
	for i in range(buffers.size()):
		if buffers[i]:
			rdmain.free_rid(buffers[i])
	if shader:
		rdmain.free_rid(shader)
	# TODO: consider other RIDs

func restart_simulation():
	# Use startup settings
	agent_count = start_agent_count
	if (species_count != start_species_count && !%CheckBoxLockMatrix.disabled && %CheckBoxLockMatrix.button_pressed):
		%CheckBoxLockMatrix.button_pressed = false
	species_count = start_species_count

	# Create playfield
	var start_data : Dictionary = {}
	match starting_method:
		0: start_data = StartupManager.build_particles(self, StartupManager.pos_random, false)
		1: start_data = StartupManager.build_particles(self, StartupManager.pos_random, true)
		
		2: start_data = StartupManager.build_particles(self, StartupManager.pos_ring, false)
		3: start_data = StartupManager.build_particles(self, StartupManager.pos_ring, true)
		
		4: 
			StartupManager.setup_spiral_params(self)
			start_data = StartupManager.build_particles(self, StartupManager.pos_spiral, false)
		5: 
			StartupManager.setup_spiral_params(self)
			start_data = StartupManager.build_particles(self, StartupManager.pos_spiral, true)
			
		6: start_data = StartupManager.build_particles(self, StartupManager.pos_columns, false)
		7: start_data = StartupManager.build_particles(self, StartupManager.pos_columns, true)
		
		_: start_data = StartupManager.build_particles(self, StartupManager.pos_random, false)
	rebuild_buffers(start_data)

	# Unlock Checkbox
	%CheckBoxLockMatrix.disabled = false 

func rebuild_buffers(data: Dictionary):
	buffers.clear()
	uniforms.clear()

	var pos_bytes :PackedByteArray= data["pos"].to_byte_array()
	var vel_bytes :PackedByteArray= data["vel"].to_byte_array()
	var species_bytes :PackedByteArray= data["species"].to_byte_array()
	var interaction_bytes :PackedByteArray= data["interaction_matrix"].to_byte_array()

	# IN BUFFERS
	buffers.append(rdmain.storage_buffer_create(pos_bytes.size(), pos_bytes))      # 0
	buffers.append(rdmain.storage_buffer_create(vel_bytes.size(), vel_bytes))      # 1
	buffers.append(rdmain.storage_buffer_create(species_bytes.size(), species_bytes))  # 2

	# OUT BUFFERS (copy of input to start)
	for b in [pos_bytes, vel_bytes]:
		buffers.append(rdmain.storage_buffer_create(b.size(), b))  # 3, 4

	# Interaction Matrix
	buffers.append(rdmain.storage_buffer_create(interaction_bytes.size(), interaction_bytes))  # 5

	# === COLLISION BUFFERS ===
	# One per agent (collision counts)
	var count_bytes := PackedByteArray()
	count_bytes.resize(int(agent_count) * 4) # 4 bytes per uint (zero-filled)
	buffers.append(rdmain.storage_buffer_create(count_bytes.size(), count_bytes))  # 6 CollisionCountBuffer
	# agent_count * MAX_COLLISIONS (partner indices)
	var partners_bytes := PackedByteArray()
	partners_bytes.resize(int(agent_count) * int(MAX_COLLISIONS) * 4)
	buffers.append(rdmain.storage_buffer_create(partners_bytes.size(), partners_bytes))  # 7 CollisionPartnerBuffer

	# === SPATIAL HASIHNG BUFFERS ===
	# Compute Number of Cells
	var world_size := float(image_size) * float(world_size_mult) # same as GLSL's world
	cell_size = 500 #max(boid_vision_radius, species_interaction_radius, collision_radius)
	cells_per_row = int(ceil(world_size / cell_size))
	#cells_per_row = 46# hardcode
	#print(cells_per_row)
	num_cells = cells_per_row * cells_per_row
	#print(num_cells)
	# Cell counts buffer (per cell)
	var cell_counts_b := PackedByteArray()
	cell_counts_b.resize(num_cells * 4)
	buffers.append(rdmain.storage_buffer_create(cell_counts_b.size(), cell_counts_b))  # binding 8
	# Cell offsets buffer (per cell)
	var cell_offsets_b := PackedByteArray()
	cell_offsets_b.resize(num_cells * 4)
	buffers.append(rdmain.storage_buffer_create(cell_offsets_b.size(), cell_offsets_b))  # binding 9
	# Sorted indices (per agent)
	var sorted_indices_b := PackedByteArray()
	sorted_indices_b.resize(int(agent_count) * 4)
	buffers.append(rdmain.storage_buffer_create(sorted_indices_b.size(), sorted_indices_b))  # binding 10
	# Agent -> cell mapping (per agent)
	var agent_cell_b := PackedByteArray()
	agent_cell_b.resize(int(agent_count) * 4)
	buffers.append(rdmain.storage_buffer_create(agent_cell_b.size(), agent_cell_b))  # binding 11
	# Cursor per cell (per cell)
	var cursor_b := PackedByteArray()
	cursor_b.resize(num_cells * 4)
	buffers.append(rdmain.storage_buffer_create(cursor_b.size(), cursor_b))  # binding 12

	# Output texture
	var output_img := Image.create(image_size, image_size, false, Image.FORMAT_RGBAF)
	#texture = ImageTexture.create_from_image(output_img)
	output_tex = rdmain.texture_create(fmt, view, [output_img.get_data()])
	textureRD.texture_rd_rid = output_tex

	# UNIFORMS
	for i in range(13):
		var u := RDUniform.new()
		u.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		u.binding = i
		u.add_id(buffers[i])
		uniforms.append(u)

	# IMAGE TEXTURE OUTPUT
	output_tex_uniform = RDUniform.new()
	output_tex_uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	output_tex_uniform.binding = 13
	output_tex_uniform.add_id(output_tex)
	uniforms.append(output_tex_uniform)

	# SHADER + PIPELINE
	var shader_file := load("res://particle_boids.glsl") as RDShaderFile
	shader = rdmain.shader_create_from_spirv(shader_file.get_spirv())
	pipeline = rdmain.compute_pipeline_create(shader)
	uniform_set = rdmain.uniform_set_create(uniforms, shader, 0)

func compute_stage(run_mode:int):
	var global_size_x : int = int(ceil(float(agent_count) / shader_local_size)) + 1 # per-agent pass
	var global_size_y : int = 1
	
	if (run_mode == 3): # draw grid by pixel
		global_size_x = image_size*image_size # per pixel 1D pass
	
	#global_size_x = image_size # per pixel 2D pass
	#global_size_y = image_size # per pixel 2D pass
	
	#global_size_x = int(ceil(float(num_cells) / shader_local_size)) + 1 # per-cell pass
	
	var compute_list := rdmain.compute_list_begin()
	rdmain.compute_list_bind_compute_pipeline(compute_list, pipeline)
	rdmain.compute_list_bind_uniform_set(compute_list, uniform_set, 0)

	# PUSH CONSTANT PARAMETERS
	var params := PackedFloat32Array([
		run_mode,
		dt,
		mix_t,
		agent_count,
		species_count,
		
		boid_vision_radius,
		species_interaction_radius,
		
		alignment_force,
		cohesion_force,
		separation_force,
		
		movement_randomness,
		movement_scaling,
		force_softening,
		center_attraction,
		damping,
		min_speed,
		max_speed,
		max_force,
		
		collision_radius,
		MAX_COLLISIONS,
		cell_size,
		cells_per_row,
		
		draw_radius,
		image_size,
		world_size_mult,
		camera_center.x,
		camera_center.y,
		zoom,
		#0.0,0.0,0.0
	])
	var params_bytes := PackedByteArray()
	params_bytes.append_array(params.to_byte_array())

	rdmain.compute_list_set_push_constant(compute_list, params_bytes, params_bytes.size())
	rdmain.compute_list_dispatch(compute_list, global_size_x, global_size_y, 1) 
	rdmain.compute_list_end()
	#rdmain.submit()
	#rdmain.sync()

func _process(_delta):
	RenderingServer.call_on_render_thread(run_simulation)

func run_simulation():
	# ---------- SPATIAL HASHING PASSES ----------
	
	# zero cell counts
	var empty_counts_bytes :PackedByteArray
	empty_counts_bytes.resize(num_cells * 4)
	rdmain.buffer_update(buffers[8], 0, empty_counts_bytes.size(), empty_counts_bytes)

	# zero collide counts
	var empty_collide_counts_bytes :PackedByteArray
	empty_collide_counts_bytes.resize(agent_count * 4)
	rdmain.buffer_update(buffers[6], 0, empty_collide_counts_bytes.size(), empty_collide_counts_bytes)
	
	# count cells (agents per cell)
	compute_stage(10)  

	# compute prefix sum
	var cell_counts_bytes = rdmain.buffer_get_data(buffers[8])
	var counts : PackedInt32Array = cell_counts_bytes.to_int32_array()
	var offsets = PackedInt32Array()
	offsets.resize(num_cells)
	var running := 0
	for i in range(num_cells):
		offsets[i] = running
		running += counts[i]

	# upload offsets AND cursor
	var offsets_bytes : PackedByteArray = offsets.to_byte_array()
	rdmain.buffer_update(buffers[9], 0, offsets_bytes.size(), offsets_bytes) # cell_offsets
	rdmain.buffer_update(buffers[12], 0, offsets_bytes.size(), offsets_bytes) # cursor
	
	# scatter sorted indices
	compute_stage(11)  
	
	# ---------- SIMULATION PASSES ----------
	
	# run simulation + gather collisions
	compute_stage(0) 
	
	# collision resolution
	compute_stage(1)  
	
	# ---------- RENDER PASSES ----------
	
	# clear
	rdmain.texture_update(output_tex, 0, empty_img.get_data())
	
	# draw
	compute_stage(2)  
	
	# (optional) draw partitioning grid
	if(%CheckShowGrid.button_pressed):
		compute_stage(3)  
	
	### RESOLVE RESULTS — swap the output to be the input for next frame
	swap_buffer_bindings()
	
	# --- Update texture ---
	texture = textureRD

# FRAME BUFFER SWAP LOGIC
var swap_flag : int = 0
func swap_buffer_bindings():
	if (dt == 0):
		return
	
	swap_flag = 1 - swap_flag  # toggle between 0 and 1

	var pos_in_index = 0
	var vel_in_index = 0
	var pos_out_index = 0
	var vel_out_index = 0
	if swap_flag == 0:
		# Use first set as input, second set as output
		pos_in_index = 0
		vel_in_index = 1
		pos_out_index = 3
		vel_out_index = 4
	else:
		# Use second set as input, first set as output
		pos_in_index = 3
		vel_in_index = 4
		pos_out_index = 0
		vel_out_index = 1
	
	# rebuild uniforms for bindings 0 and 1
	uniforms[0] = RDUniform.new()
	uniforms[0].uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	uniforms[0].binding = 0
	uniforms[0].add_id(buffers[pos_in_index])
	uniforms[1] = RDUniform.new()
	uniforms[1].uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	uniforms[1].binding = 1
	uniforms[1].add_id(buffers[vel_in_index])
	
	uniforms[3] = RDUniform.new()
	uniforms[3].uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	uniforms[3].binding = 3
	uniforms[3].add_id(buffers[pos_out_index])
	uniforms[4] = RDUniform.new()
	uniforms[4].uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	uniforms[4].binding = 4
	uniforms[4].add_id(buffers[vel_out_index])
	
	# rebuild uniform set
	rdmain.free_rid(uniform_set)
	uniform_set = rdmain.uniform_set_create(uniforms, shader, 0)

# HANDLE MOUSE INPUTS
var dragging := false
var last_mouse_pos := Vector2()
func _gui_input(event):
	if event is InputEventMouseButton:
		# Handle zoom
		if event.button_index == MOUSE_BUTTON_WHEEL_UP:
			zoom = clamp(zoom * 1.05, MIN_ZOOM, MAX_ZOOM)
		elif event.button_index == MOUSE_BUTTON_WHEEL_DOWN:
			zoom = clamp(zoom / 1.05, MIN_ZOOM, MAX_ZOOM)

		# Start/stop panning with right mouse button
		elif event.button_index == MOUSE_BUTTON_RIGHT:
			dragging = event.pressed
			last_mouse_pos = event.position

	elif event is InputEventMouseMotion and dragging:
		# Convert drag delta to world space based on zoom
		var delta :Vector2= (event.position - last_mouse_pos) / zoom
		last_mouse_pos = event.position
		camera_center -= delta
