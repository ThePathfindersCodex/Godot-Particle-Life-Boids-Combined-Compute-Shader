extends TextureRect

# CONFIG
var compute_texture_size :int= 256 # Holds up to 256*256 pixel particles
var viewport_size :int= 800
var shader_local_size_x := 16
var shader_local_size_y := 16
@onready var image_size = compute_texture_size
var zone_size_mult : int = 20 # wrap around border world size
var agent_count : int = 1024*15
var species_count : int = 8

# STARTUP PARAMS
var starting_method : int = 4 # method to use when restarting new field?
var rand_start_interaction_range : float = 2.0 # force will be random between -X and +X
var rand_start_radius_mul : float = 2.0 # different startup patterns use this multiplier
var start_agent_count : int = agent_count # only used when restarting new field
var start_species_count : int = species_count # only used when restarting new field

# SPEED/TIME
var dt : float = .25
var paused_dt : float = dt # only used for pause/resume feature

# MIX SIMS
var mix_t: float = 0.5 # [0.0 == full boids; == 1.0 full particle life]

# VISION KERNELS
var boid_vision_radius : float = 200.0 # 350.0
var species_interaction_radius : float = 150.0 # 250.0

# BOIDS PARAMETERS
var alignment_force : float = 1.0
var cohesion_force : float = 1.0
var separation_force : float = 1.0

# FORCE ADJUSTMENTS
var movement_randomness : float = 0.01
var movement_scaling : float = 1.0
var force_softening_mul : float = 1.0 # 0.1 # 3.0
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
var collision_radius : float = 4.0

# CAMERA
var camera_center : Vector2 = Vector2.ZERO
var zoom : float = 0.82 # 0.5
const MIN_ZOOM := 0.1
const MAX_ZOOM := 5.0

# SPATIAL HASHING
var cell_size : int = 500
var cells_per_row : int = 0
var num_cells : int = 0

# INTERACTION MATRIX
var interaction_matrix : PackedFloat32Array = []

# RENDERER SETUP
var rdmain := RenderingServer.get_rendering_device()
var textureRD: Texture2DRD
var shader : RID
var pipeline : RID

var fmt := RDTextureFormat.new()
var view := RDTextureView.new()
var buffers : Array[RID] = []

var input_particles : RID
var output_particles : RID
var multimesh := MultiMesh.new()
var quadmesh := QuadMesh.new()
var render_material := ShaderMaterial.new()

func _ready():
	randomize()
	fmt.width = compute_texture_size
	fmt.height = compute_texture_size
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
	#if textureRD:
		#textureRD.texture_rd_rid = RID()
	RenderingServer.call_on_render_thread(_free_compute_resources)

func _free_compute_resources():
	if textureRD:
		textureRD.texture_rd_rid = RID()
	for i in range(buffers.size()):
		if buffers[i]:
			rdmain.free_rid(buffers[i])
	if input_particles:
		rdmain.free_rid(input_particles)
	if output_particles:
		rdmain.free_rid(output_particles)
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
	_free_compute_resources()
	buffers.clear()

	var img_particles := Image.create(
		compute_texture_size,
		compute_texture_size,
		false,
		Image.FORMAT_RGBAF
	)

	for i in agent_count:
		var x :int= i % compute_texture_size
		@warning_ignore("integer_division")
		var y :int= i / compute_texture_size
		img_particles.set_pixel(
			x, y,
			Color(data["pos"][i].x, data["pos"][i].y, data["vel"][i].x, data["vel"][i].y)
		)
	var data_particles := img_particles.get_data()
	input_particles = rdmain.texture_create(fmt, view, [data_particles])
	output_particles = rdmain.texture_create(fmt, view, [data_particles])

	# Interaction Matrix
	var species_bytes :PackedByteArray= data["species"].to_byte_array()
	var interaction_bytes :PackedByteArray= data["interaction_matrix"].to_byte_array()
	buffers.append(rdmain.storage_buffer_create(species_bytes.size(), species_bytes))
	buffers.append(rdmain.storage_buffer_create(interaction_bytes.size(), interaction_bytes))

	# === COLLISION BUFFERS ===
	# One per agent (collision counts)
	var count_bytes := PackedByteArray()
	count_bytes.resize(int(agent_count) * 4) # 4 bytes per uint (zero-filled)
	buffers.append(rdmain.storage_buffer_create(count_bytes.size(), count_bytes))  # CollisionCountBuffer
	# agent_count * MAX_COLLISIONS (partner indices)
	var partners_bytes := PackedByteArray()
	partners_bytes.resize(int(agent_count) * int(MAX_COLLISIONS) * 4)
	buffers.append(rdmain.storage_buffer_create(partners_bytes.size(), partners_bytes))  # CollisionPartnerBuffer

	# === SPATIAL HASIHNG BUFFERS ===
	# Compute Number of Cells
	var world_size := float(image_size) * float(zone_size_mult) # same as GLSL's world
	cells_per_row = int(ceil(world_size / cell_size))
	#print(cells_per_row)
	num_cells = cells_per_row * cells_per_row
	#print(num_cells)
	# Cell counts buffer (per cell)
	var cell_counts_b := PackedByteArray()
	cell_counts_b.resize(num_cells * 4)
	buffers.append(rdmain.storage_buffer_create(cell_counts_b.size(), cell_counts_b))
	# Cell offsets buffer (per cell)
	var cell_offsets_b := PackedByteArray()
	cell_offsets_b.resize(num_cells * 4)
	buffers.append(rdmain.storage_buffer_create(cell_offsets_b.size(), cell_offsets_b))
	# Sorted indices (per agent)
	var sorted_indices_b := PackedByteArray()
	sorted_indices_b.resize(int(agent_count) * 4)
	buffers.append(rdmain.storage_buffer_create(sorted_indices_b.size(), sorted_indices_b))
	# Agent -> cell mapping (per agent)
	var agent_cell_b := PackedByteArray()
	agent_cell_b.resize(int(agent_count) * 4)
	buffers.append(rdmain.storage_buffer_create(agent_cell_b.size(), agent_cell_b))
	# Cursor per cell (per cell)
	var cursor_b := PackedByteArray()
	cursor_b.resize(num_cells * 4)
	buffers.append(rdmain.storage_buffer_create(cursor_b.size(), cursor_b))

	# Output texture
	textureRD.texture_rd_rid = output_particles

	# multimesh/instance/mesh/material
	#var mask :Texture2D= load("res://triangle.png")
	var mask :GradientTexture2D= load("res://my_circle.tres")
	render_material.shader = load("res://particle_draw.gdshader")
	render_material.set_shader_parameter("alpha_tex", mask)
	render_material.set_shader_parameter("particle_buffer", textureRD)
	var heatmap_colors :GradientTexture1D= load("res://my_gradient_heatmap.tres")
	render_material.set_shader_parameter("gradient_texture", heatmap_colors)
	render_material.set_shader_parameter("species_count", species_count)
	render_material.set_shader_parameter("camera_center", camera_center)
	render_material.set_shader_parameter("zoom", zoom)
	render_material.set_shader_parameter("compute_texture_size", compute_texture_size)
	render_material.set_shader_parameter("viewport_size", Vector2(viewport_size, viewport_size))
	%MMI.material = render_material # 2D

	quadmesh.size = Vector2.ONE
	multimesh.instance_count = 0 # can only set other values when instance_count==0
	multimesh.mesh = quadmesh
	multimesh.transform_format = MultiMesh.TRANSFORM_2D
	multimesh.use_colors = false
	multimesh.use_custom_data = true
	multimesh.instance_count = agent_count # actual point count
	for i in range(agent_count):
		multimesh.set_instance_transform_2d(i, Transform2D())
		multimesh.set_instance_custom_data(i, Color(data["species"][i],0,0,0))

	%MMI.multimesh = multimesh

	# SHADER + PIPELINE
	var shader_file := load("res://particle_boids.glsl") as RDShaderFile
	shader = rdmain.shader_create_from_spirv(shader_file.get_spirv())
	pipeline = rdmain.compute_pipeline_create(shader)

func compute_stage(run_mode:int,input_set,output_set):
	var global_size_x : int
	var global_size_y : int
	
	var group_size = shader_local_size_x * shader_local_size_y # 16*16 = 256
	
	# --- texture based passes ---
	if run_mode in [0,1,10]:
		global_size_x = int(ceil(float(compute_texture_size) / shader_local_size_x))
		global_size_y = int(ceil(float(compute_texture_size) / shader_local_size_y))

	# --- prefix scan ---
	elif run_mode == 11:
		global_size_x = int(ceil(float(num_cells) / float(group_size)))
		global_size_y = 1

	# --- scatter ---
	elif run_mode == 12:
		global_size_x = int(ceil(float(agent_count) / float(group_size)))
		global_size_y = 1

	var compute_list := rdmain.compute_list_begin()
	rdmain.compute_list_bind_compute_pipeline(compute_list, pipeline)
	rdmain.compute_list_bind_uniform_set(compute_list, input_set, 0)
	rdmain.compute_list_bind_uniform_set(compute_list, output_set, 1)

	# PUSH CONSTANT PARAMETERS
	var params := PackedFloat32Array([
		run_mode,
		dt,
		compute_texture_size,
		mix_t,
		float(agent_count),
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
		
		float(image_size),
		float(zone_size_mult),
		0.0,0.0,0.0
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
	# Flip buffers via uniformsets
	var frame_flip = flip_buffers()
	var input_set  = frame_flip[0]
	var output_set = frame_flip[1]
	
	# ---------- SPATIAL HASHING PASSES ----------
	
	# zero cell counts
	var empty_counts_bytes :PackedByteArray
	empty_counts_bytes.resize(num_cells * 4)
	rdmain.buffer_update(buffers[4], 0, empty_counts_bytes.size(), empty_counts_bytes)

	# zero collide counts
	var empty_collide_counts_bytes :PackedByteArray
	empty_collide_counts_bytes.resize(agent_count * 4)
	rdmain.buffer_update(buffers[2], 0, empty_collide_counts_bytes.size(), empty_collide_counts_bytes)
	
	# count cells (agents per cell)
	compute_stage(10,input_set,output_set)  
	# compute prefix sum
	compute_stage(11,input_set,output_set)  
	# scatter sorted indices
	compute_stage(12,input_set,output_set)

	# ---------- SIMULATION PASSES ----------
	
	# run simulation + gather collisions
	compute_stage(0,input_set,output_set) 
	
	# collision resolution
	compute_stage(1,input_set,output_set)  
	
	# UPDATE MATERIAL BUFFERS
	render_material.set_shader_parameter("particle_buffer", textureRD)
	render_material.set_shader_parameter("camera_center", camera_center)
	render_material.set_shader_parameter("zoom", zoom)

	rdmain.free_rid(input_set)
	rdmain.free_rid(output_set)

var ping : bool = false
func flip_buffers():
	# Flip buffers
	ping = !ping
	var read_main  : RID
	var read_sec   : RID
	var write_main : RID
	var write_sec  : RID
	if ping:
		read_main  = output_particles
		write_main = input_particles
	else:
		read_main  = input_particles
		write_main = output_particles

	# use correct output image
	if textureRD:
		textureRD.texture_rd_rid = write_main
	
	# Create uniform sets
	var input_set  := _create_uniform_set(read_main,  read_sec,  0)
	var output_set := _create_uniform_set(write_main, write_sec, 1)
	
	return [input_set,output_set]

func _create_uniform_set(texture_rd: RID, texture_rd2: RID, _uniform_set: int) -> RID:
	var uniform := RDUniform.new()
	uniform.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	uniform.binding = 0
	uniform.add_id(texture_rd)
	
	var uniform2 := RDUniform.new()
	uniform2.uniform_type = RenderingDevice.UNIFORM_TYPE_IMAGE
	uniform2.binding = 0
	uniform2.add_id(texture_rd2)
	
	var uniform3 := RDUniform.new()
	uniform3.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	uniform3.binding = 1
	uniform3.add_id(buffers[0]) #  in_species_buffer

	var uniform4 := RDUniform.new()
	uniform4.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	uniform4.binding = 2
	uniform4.add_id(buffers[1]) # interaction_matrix
	
	var uniform5 := RDUniform.new()
	uniform5.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	uniform5.binding = 3
	uniform5.add_id(buffers[2]) # 3  collision_count_buffer
	
	var uniform6 := RDUniform.new()
	uniform6.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	uniform6.binding = 4
	uniform6.add_id(buffers[3]) # 4  collision_partner_buffer
	
	var uniform7 := RDUniform.new()
	uniform7.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	uniform7.binding = 5
	uniform7.add_id(buffers[4]) # Cell counts buffer
	
	var uniform8 := RDUniform.new()
	uniform8.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	uniform8.binding = 6
	uniform8.add_id(buffers[5]) # Cell offsets buffer
	
	var uniform9 := RDUniform.new()
	uniform9.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	uniform9.binding = 7
	uniform9.add_id(buffers[6]) # Sorted indices
	
	var uniform10 := RDUniform.new()
	uniform10.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	uniform10.binding = 8
	uniform10.add_id(buffers[7]) # Agent -> cell mapping
	
	var uniform11 := RDUniform.new()
	uniform11.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
	uniform11.binding = 9
	uniform11.add_id(buffers[8]) # Cursor per cell
	
	var new_set = [uniform, uniform2, uniform3, uniform4, uniform5, uniform6, uniform7, uniform8, uniform9, uniform10, uniform11]
	
	return rdmain.uniform_set_create(new_set, shader, _uniform_set)
