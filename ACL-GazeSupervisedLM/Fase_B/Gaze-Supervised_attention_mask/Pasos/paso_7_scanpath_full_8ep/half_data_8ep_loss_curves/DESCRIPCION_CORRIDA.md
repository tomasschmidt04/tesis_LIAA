# BETO + scanpath half-data pretraining

Generated at: `2026-05-03T19:36:22-03:00`

## Descripcion

Corrida de preentrenamiento MLM de Paso 7 usando aproximadamente la mitad de los datos de las corridas full previas. Los artefactos quedan dentro de la carpeta de la arquitectura de referencia para mantener juntos checkpoints, curvas y resultados.

## Arquitectura de referencia

- `Pasos/paso_7_scanpath_full_8ep`

## Hiperparametros

- model_name_or_path: `dccuchile/bert-base-spanish-wwm-cased`
- measured_scanpath_file: `/home/tomi/tesis_LIAA/reading-et/aligned_output/aligned_scanpaths.jsonl`
- split: `train`
- max_train_samples: `31836`
- max_eval_samples: `1000`
- num_train_epochs: `8`
- max_seq_length: `128`
- per_device_train_batch_size: `4`
- per_device_eval_batch_size: `4`
- learning_rate: `5e-05`
- max_masked_positions: `3`
- seed: `13`
- aux_weight: `0.1`
- save_every_epoch: `True` unless overridden by extra_cli_args
- extra_cli_args: `none`

## Artefactos principales

- output_dir: `Pasos/paso_7_scanpath_full_8ep/half_data_8ep_loss_curves`
- loss_curves_csv: `Pasos/paso_7_scanpath_full_8ep/half_data_8ep_loss_curves/loss_curves.csv`
- loss_plots_dir: `Pasos/paso_7_scanpath_full_8ep/half_data_8ep_loss_curves/loss_plots`
- checkpoint_final: `Pasos/paso_7_scanpath_full_8ep/half_data_8ep_loss_curves/checkpoint_final`
- best_checkpoint: `Pasos/paso_7_scanpath_full_8ep/half_data_8ep_loss_curves/best_checkpoint`
