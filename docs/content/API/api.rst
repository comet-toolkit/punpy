.. currentmodule:: punpy

.. _api:

#############
API reference
#############

This page provides an auto-generated summary of **punpy**'s API. For more details
and examples, refer to the relevant chapters in the main part of the
documentation.

MCPropagation
====================

.. autosummary::
   :toctree: generated/

   mc.mc_propagation.MCPropagation
   mc.mc_propagation.MCPropagation.propagate_standard
   mc.mc_propagation.MCPropagation.propagate_random
   mc.mc_propagation.MCPropagation.propagate_systematic
   mc.mc_propagation.MCPropagation.propagate_cov
   mc.mc_propagation.MCPropagation.generate_MC_sample
   mc.mc_propagation.MCPropagation.generate_MC_sample_cov
   mc.mc_propagation.MCPropagation.propagate_cov_flattened
   mc.mc_propagation.MCPropagation.run_samples
   mc.mc_propagation.MCPropagation.combine_samples
   mc.mc_propagation.MCPropagation.process_samples

LPUPropagation
=====================

.. autosummary::
   :toctree: generated/

   lpu.lpu_propagation.LPUPropagation
   lpu.lpu_propagation.LPUPropagation.propagate_standard
   lpu.lpu_propagation.LPUPropagation.propagate_random
   lpu.lpu_propagation.LPUPropagation.propagate_systematic
   lpu.lpu_propagation.LPUPropagation.propagate_cov
   lpu.lpu_propagation.LPUPropagation.propagate_flattened_cov
   lpu.lpu_propagation.LPUPropagation.process_jacobian

Digital Effects Tables
=======================

.. autosummary::
   :toctree: generated/

   digital_effects_table.measurement_function.MeasurementFunction
   digital_effects_table.measurement_function.MeasurementFunction.meas_function
   digital_effects_table.measurement_function.MeasurementFunction.get_argument_names
   digital_effects_table.measurement_function.MeasurementFunction.get_measurand_name_and_unit
   digital_effects_table.measurement_function.MeasurementFunction.update_measurand
   digital_effects_table.measurement_function.MeasurementFunction.setup
   digital_effects_table.measurement_function.MeasurementFunction.propagate_ds
   digital_effects_table.measurement_function.MeasurementFunction.propagate_ds_total
   digital_effects_table.measurement_function.MeasurementFunction.propagate_ds_specific
   digital_effects_table.measurement_function.MeasurementFunction.propagate_ds_all
   digital_effects_table.measurement_function.MeasurementFunction.run
   digital_effects_table.measurement_function.MeasurementFunction.propagate_total
   digital_effects_table.measurement_function.MeasurementFunction.propagate_random
   digital_effects_table.measurement_function.MeasurementFunction.propagate_systematic
   digital_effects_table.measurement_function.MeasurementFunction.propagate_structured
   digital_effects_table.measurement_function.MeasurementFunction.propagate_specific

   digital_effects_table.measurement_function_utils.MeasurementFunctionUtils
   digital_effects_table.measurement_function_utils.MeasurementFunctionUtils.find_comps
   digital_effects_table.measurement_function_utils.MeasurementFunctionUtils.get_input_qty
   digital_effects_table.measurement_function_utils.MeasurementFunctionUtils.get_input_unc
   digital_effects_table.measurement_function_utils.MeasurementFunctionUtils.calculate_unc
   digital_effects_table.measurement_function_utils.MeasurementFunctionUtils.calculate_unc_missingdim
   digital_effects_table.measurement_function_utils.MeasurementFunctionUtils.get_input_corr
   digital_effects_table.measurement_function_utils.MeasurementFunctionUtils.calculate_corr
   digital_effects_table.measurement_function_utils.MeasurementFunctionUtils.calculate_corr_missingdim

   digital_effects_table.digital_effects_table_templates.DigitalEffectsTableTemplates
   digital_effects_table.digital_effects_table_templates.DigitalEffectsTableTemplates.make_template_main
   digital_effects_table.digital_effects_table_templates.DigitalEffectsTableTemplates.make_template_tot
   digital_effects_table.digital_effects_table_templates.DigitalEffectsTableTemplates.make_template_specific
   digital_effects_table.digital_effects_table_templates.DigitalEffectsTableTemplates.remove_unc_component
   digital_effects_table.digital_effects_table_templates.DigitalEffectsTableTemplates.join_with_preexisting_ds
