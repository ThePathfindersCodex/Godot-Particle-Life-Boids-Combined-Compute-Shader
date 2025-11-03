extends Node2D

var slider_bindings = {
	"%SliderSimMix": "mix_t",
	"%SliderSpeed": "dt",
	"%SliderDampen": "damping",
	"%SliderSenseRadius": "species_interaction_radius",
	"%SliderForceSoftenMultiplier": "force_softening_mul",
	"%SliderDrawSize": "draw_radius",
	"%SliderCollideModifier": "collision_modifier",
	"%SliderCenterPull": "center_attraction",
	"%SliderMaxForce": "max_force",
	"%SliderVisionRadius": "boid_vision_radius",
	"%SliderAlignmentForce": "alignment_force",
	"%SliderCohesionForce": "cohesion_force",
	"%SliderSeparationForce": "separation_force",
	"%SliderMinSpeed": "min_speed",
	"%SliderMaxSpeed": "max_speed",
	"%SliderMovementRandomness": "movement_randomness",
	"%SliderMovementScaling": "movement_scaling",
}


func _ready():
	# Connect all sliders and set defaults
	for slider_path in slider_bindings.keys():
		var slider = get_node(slider_path)
		var property = slider_bindings[slider_path]
		slider.value = %ParticleBoids.get(property)
		slider.connect("value_changed", Callable(self, "_on_slider_changed").bind(property))
	
	## Option dropdown default values
	%OptionStartInteractionRange.selected = int(%ParticleBoids.rand_start_interaction_range)
	%OptionStartRadiusMultiplier.selected =  int(%ParticleBoids.rand_start_radius_mul)
	%OptionStartSpeciesCount.selected = %ParticleBoids.start_species_count
	%OptionStartMethod.selected = %ParticleBoids.starting_method
	%OptionStartPointCount.selected = 4
	
func _process(_delta):
	## SET READOUT VALUES
	%LabelPointsValue.text = str(snapped(%ParticleBoids.agent_count,1))
	%LabelSpeciesValue.text = str(snapped(%ParticleBoids.species_count,1))
	%LabelSpeedValue.text = str(snapped(%ParticleBoids.dt,.01))
	%LabelDampingValue.text = str(snapped(%ParticleBoids.damping,.01))
	%LabelSenseRadiusValue.text = str(snapped(%ParticleBoids.species_interaction_radius,.01))
	%LabelForceSoftenMultiplierValue.text = str(snapped(%ParticleBoids.force_softening_mul,.1))
	%LabelForceSoftenValue.text = str(snapped(%ParticleBoids.force_softening,.01))
	%LabelDrawSizeValue.text = str(snapped(%ParticleBoids.draw_radius,.01))
	%LabelCollideModifierValue.text = str(snapped(%ParticleBoids.collision_modifier,.1))
	%LabelCollideRadiusValue.text = str(snapped(%ParticleBoids.collision_radius,.01))
	%LabelCenterPullValue.text = str(snapped(%ParticleBoids.center_attraction,.01))
	%LabelMaxForceValue.text = str(snapped(%ParticleBoids.max_force,.01))
	%LabelCamCenterValue.text = "("+str(snapped(%ParticleBoids.camera_center.x,.1))+ ", " + str(snapped(%ParticleBoids.camera_center.y,.1)) + ")"
	%LabelZoomValue.text = str(snapped(%ParticleBoids.zoom,.01))
	%LabelVisionRadiusValue.text = str(snapped(%ParticleBoids.boid_vision_radius,1))
	%LabelAlignmentForceValue.text = str(snapped(%ParticleBoids.alignment_force,.1))
	%LabelCohesionForceValue.text = str(snapped(%ParticleBoids.cohesion_force,.1))
	%LabelSeparationForceValue.text = str(snapped(%ParticleBoids.separation_force,.1))
	%LabelMinSpeedValue.text = str(snapped(%ParticleBoids.min_speed,.1))
	%LabelMaxSpeedValue.text = str(snapped(%ParticleBoids.max_speed,.1))
	%LabelSliderMovementRandomnessValue.text = str(snapped(%ParticleBoids.movement_randomness,.01))
	%LabelSliderMovementScalingValue.text = str(snapped(%ParticleBoids.movement_scaling,.1))
	
	# SET INTERACTION MATRIX TEXT
	var species_count = %ParticleBoids.species_count
	var interaction_matrix = %ParticleBoids.interaction_matrix
	var lines := []
	for i in range(species_count):
		var line = str(i + 1) + ": "
		for j in range(species_count):
			var val = snapped(interaction_matrix[i * species_count + j], 0.01)
			line += "%5.2f" % val
			if j < species_count - 1:
				line += " | "
		lines.append(line)
	%LabelIntMatrix.text = "INTERACTION MATRIX\n" + "\n".join(lines)

	# HANDLE PAUSE/RESUME
	if Input.is_action_just_pressed("pause_resume"):
		pause_resume()

func pause_resume():
	if %ParticleBoids.dt == 0.0:
		%ParticleBoids.dt = %ParticleBoids.paused_dt  # resume
		%ButtonPauseResume.text = "PAUSE"
	else:
		%ParticleBoids.paused_dt = %ParticleBoids.dt  # store current
		%ParticleBoids.dt = 0.0       # pause
		%ButtonPauseResume.text = "RESUME"

func _on_slider_changed(value: float, property: String):
	%ParticleBoids.set(property, value)

func _on_button_restart_pressed() -> void:
	%ParticleBoids.restart_simulation()

func _on_button_pause_resume_pressed() -> void:
	pause_resume()

func _on_option_start_method_item_selected(index: int) -> void:
	%ParticleBoids.starting_method=index

func _on_option_start_interaction_range_item_selected(index: int) -> void:
	%ParticleBoids.rand_start_interaction_range=index

func _on_option_start_radius_multiplier_item_selected(index: int) -> void:
	%ParticleBoids.rand_start_radius_mul=index * 2

func _on_option_start_point_count_item_selected(index: int) -> void:
	%ParticleBoids.start_agent_count=int(%OptionStartPointCount.get_item_text(index))

func _on_option_start_species_count_item_selected(index: int) -> void:
	%ParticleBoids.start_species_count=index
	
	%CheckBoxLockMatrix.disabled=false
	if (%ParticleBoids.start_species_count != %ParticleBoids.species_count):
			%CheckBoxLockMatrix.disabled=true
