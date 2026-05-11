import argparse
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from Gazesup_bert_combined_mlm_model import Gazesup_BERTForCombinedMaskedLM


DEFAULT_SOURCE_CHECKPOINT = "Pasos/paso_7/best_checkpoint"
DEFAULT_OUTPUT_DIR = "Pasos/paso_8"
DEFAULT_EXPORT_DIRNAME = "export_backbone_hf"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export and verify the reusable BERT backbone from a combined MLM + scanpath checkpoint."
    )
    parser.add_argument(
        "--source_checkpoint",
        default=DEFAULT_SOURCE_CHECKPOINT,
        help="Checkpoint produced by PASO 7 that contains the combined model.",
    )
    parser.add_argument(
        "--output_dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where paso_8 artifacts will be written.",
    )
    parser.add_argument(
        "--export_dirname",
        default=DEFAULT_EXPORT_DIRNAME,
        help="Subdirectory inside output_dir where the clean HF backbone export will be saved.",
    )
    parser.add_argument(
        "--downstream_num_labels",
        type=int,
        default=2,
        help="Number of labels used for the minimal downstream sequence-classification load test.",
    )
    return parser.parse_args()


def categorize_state_dict_keys(state_dict: Dict[str, torch.Tensor]):
    backbone_keys = sorted([key for key in state_dict if key.startswith("bert.")])
    scanpath_branch_keys = sorted([key for key in state_dict if key.startswith("sp_encoder.")])
    main_mlm_head_keys = sorted([key for key in state_dict if key.startswith("cls.")])
    auxiliary_mlm_head_keys = sorted([key for key in state_dict if key.startswith("scanpath_mlm_head.")])
    other_keys = sorted(
        [
            key for key in state_dict
            if key not in backbone_keys
            and key not in scanpath_branch_keys
            and key not in main_mlm_head_keys
            and key not in auxiliary_mlm_head_keys
        ]
    )
    return {
        "backbone_keys": backbone_keys,
        "scanpath_branch_keys": scanpath_branch_keys,
        "main_mlm_head_keys": main_mlm_head_keys,
        "auxiliary_mlm_head_keys": auxiliary_mlm_head_keys,
        "other_keys": other_keys,
    }


def summarize_key_group(group_name: str, keys: List[str], state_dict: Dict[str, torch.Tensor]) -> List[str]:
    tensor_count = len(keys)
    parameter_count = sum(int(state_dict[key].numel()) for key in keys)
    sample_keys = keys[:5]
    lines = [
        f"- {group_name}: {tensor_count} tensores | {parameter_count} parametros",
    ]
    if sample_keys:
        lines.append(f"  ejemplos: {sample_keys}")
    else:
        lines.append("  ejemplos: []")
    return lines


def inspect_and_export(args):
    source_checkpoint = Path(args.source_checkpoint)
    output_dir = Path(args.output_dir)
    export_dir = output_dir / args.export_dirname
    output_dir.mkdir(parents=True, exist_ok=True)
    export_dir.mkdir(parents=True, exist_ok=True)

    state_dict_path = source_checkpoint / "pytorch_model.bin"
    if not state_dict_path.exists():
        raise FileNotFoundError(f"Could not find pytorch_model.bin in source checkpoint: {source_checkpoint}")

    state_dict = torch.load(state_dict_path, map_location="cpu")
    grouped_keys = categorize_state_dict_keys(state_dict)

    config = AutoConfig.from_pretrained(source_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(source_checkpoint, use_fast=True)
    combined_model = Gazesup_BERTForCombinedMaskedLM.from_pretrained(source_checkpoint)

    # Export only the reusable principal BERT backbone. This intentionally excludes:
    # - the main MLM head (cls.*)
    # - the scanpath GRU branch (sp_encoder.*)
    # - the auxiliary MLM head (scanpath_mlm_head.*).
    combined_model.bert.save_pretrained(export_dir)
    tokenizer.save_pretrained(export_dir)

    exported_tokenizer = AutoTokenizer.from_pretrained(export_dir, use_fast=True)
    exported_config = AutoConfig.from_pretrained(export_dir)
    downstream_model = AutoModelForSequenceClassification.from_pretrained(
        export_dir,
        num_labels=args.downstream_num_labels,
    )

    downstream_inputs = exported_tokenizer("Prueba minima de carga downstream.", return_tensors="pt")
    with torch.no_grad():
        downstream_outputs = downstream_model(**downstream_inputs)

    return {
        "source_checkpoint": str(source_checkpoint),
        "source_model_class": combined_model.__class__.__name__,
        "source_tokenizer_class": tokenizer.__class__.__name__,
        "source_vocab_size": config.vocab_size,
        "source_model_type": config.model_type,
        "grouped_keys": grouped_keys,
        "state_dict": state_dict,
        "export_dir": str(export_dir),
        "exported_tokenizer_class": exported_tokenizer.__class__.__name__,
        "exported_config_model_type": exported_config.model_type,
        "exported_vocab_size": exported_config.vocab_size,
        "downstream_model_class": downstream_model.__class__.__name__,
        "downstream_num_labels": int(downstream_model.config.num_labels),
        "downstream_logits_shape": tuple(downstream_outputs.logits.shape),
    }


def build_debug_output(summary) -> str:
    lines: List[str] = [
        "PASO 8 - Exportar y verificar reutilizacion del backbone principal para downstream",
        "",
        "----------------------------------------",
        "Checkpoint inspection",
        "----------------------------------------",
        f"source_checkpoint: {summary['source_checkpoint']}",
        f"model_class: {summary['source_model_class']}",
        f"tokenizer: {summary['source_tokenizer_class']}",
        f"vocab_size: {summary['source_vocab_size']}",
        f"model_type: {summary['source_model_type']}",
        "",
        "Modules identificados:",
    ]
    grouped = summary["grouped_keys"]
    state_dict = summary["state_dict"]
    lines.extend(summarize_key_group("backbone principal (bert.*)", grouped["backbone_keys"], state_dict))
    lines.extend(summarize_key_group("rama auxiliar scanpath (sp_encoder.*)", grouped["scanpath_branch_keys"], state_dict))
    lines.extend(summarize_key_group("head MLM principal (cls.*)", grouped["main_mlm_head_keys"], state_dict))
    lines.extend(summarize_key_group("head MLM auxiliar (scanpath_mlm_head.*)", grouped["auxiliary_mlm_head_keys"], state_dict))
    lines.extend(summarize_key_group("otros", grouped["other_keys"], state_dict))
    lines.extend(
        [
            "",
            "----------------------------------------",
            "Export",
            "----------------------------------------",
            f"export_dir: {summary['export_dir']}",
            "archivos exportados esperados:",
            "- config.json",
            "- pytorch_model.bin o equivalente HF",
            "- tokenizer.json / tokenizer_config.json / vocab.txt / special_tokens_map.json",
            "",
            "----------------------------------------",
            "Verification",
            "----------------------------------------",
            f"tokenizer load: OK ({summary['exported_tokenizer_class']})",
            f"config load: OK (model_type={summary['exported_config_model_type']}, vocab_size={summary['exported_vocab_size']})",
            f"sequence classification model init from export: OK ({summary['downstream_model_class']})",
            f"downstream num_labels: {summary['downstream_num_labels']}",
            f"downstream logits.shape en prueba minima: {summary['downstream_logits_shape']}",
            "",
            "Que se reutiliza realmente en downstream:",
            "- solo el backbone principal BERT exportado desde bert.*",
            "",
            "Que NO se reutiliza en downstream:",
            "- sp_encoder.* (GRU / rama scanpath)",
            "- scanpath_mlm_head.* (head MLM auxiliar)",
            "- cls.* (head MLM principal de pretraining)",
            "",
            "Nota final:",
            "- este directorio queda listo para usarse como model_name_or_path en el siguiente paso downstream.",
        ]
    )
    return "\n".join(lines) + "\n"


def build_readme(args, summary, script_name: str) -> str:
    return f"""PASO 8 - README
================

Que se hizo
- Se creo un script nuevo llamado {script_name}.
- El script carga un checkpoint entrenado del paso 7.
- Inspecciona su estructura interna y separa conceptualmente:
  * backbone principal
  * rama auxiliar scanpath
  * heads MLM
- Exporta el backbone principal a un directorio limpio compatible con Hugging Face.
- Verifica que ese export puede cargarse como base de un modelo downstream estandar de clasificacion.

Que se verifico
- Que el checkpoint del paso 7 contiene un backbone principal reutilizable.
- Que el backbone principal puede separarse conceptualmente del resto de la rama auxiliar.
- Que se puede exportar un directorio limpio con pesos, config y tokenizer.
- Que ese export puede cargarse correctamente en un flujo downstream estandar con AutoModelForSequenceClassification.

Que NO se implemento todavia
- No se corrio GLUE.
- No se entreno downstream.
- No se evaluo downstream.
- No se refactorizo masivamente el repo.
- No se avanzo al paso 9.

Archivos modificados
- Pasos/README.txt

Archivos nuevos creados
- {script_name}
- Pasos/paso_8/README_paso_8.txt
- Pasos/paso_8/salida_export_backbone_debug.txt
- Pasos/paso_8/comandos_y_funciones.txt
- Pasos/paso_8/{Path(summary['export_dir']).name}/

Por que en downstream solo se reutiliza la rama principal
- El objetivo downstream posterior es usar solo el encoder principal BERT y una cabeza nueva de clasificacion o regresion.
- La rama scanpath fue util durante pretraining para inyectar supervision auxiliar, pero no forma parte del flujo estandar de inferencia downstream.
- Esto esta alineado con la idea central del paper: el modulo gaze ayuda durante entrenamiento, mientras que la prediccion downstream reutiliza el Transformer principal.

Que partes del checkpoint NO se reutilizan en downstream
- sp_encoder.*: corresponde a la rama auxiliar scanpath, incluyendo la GRU.
- scanpath_mlm_head.*: corresponde a la cabeza MLM auxiliar.
- cls.*: corresponde a la cabeza MLM principal de pretraining y no a una cabeza downstream de clasificacion/regresion.

Como usar el export resultante en el siguiente paso
- El directorio exportado queda listo para pasarse como model_name_or_path.
- En el siguiente paso downstream la idea seria usar algo como:
  --model_name_or_path {summary['export_dir']}
- Ese directorio ya fue verificado con una carga minima en AutoModelForSequenceClassification.

Parametros usados en esta corrida
- source_checkpoint = {args.source_checkpoint}
- output_dir = {args.output_dir}
- export_dirname = {args.export_dirname}
- downstream_num_labels = {args.downstream_num_labels}
"""


def build_commands_file(args, script_name: str) -> str:
    relative_output = Path(args.output_dir).as_posix()
    command = (
        f".\\.venv\\Scripts\\python.exe {script_name} "
        f"--source_checkpoint \"{args.source_checkpoint}\" "
        f"--output_dir {relative_output} "
        f"--export_dirname {args.export_dirname} "
        f"--downstream_num_labels {args.downstream_num_labels}"
    )
    return f"""COMANDOS Y FUNCIONES - PASO 8
=============================

Comando ejecutado
- {command}

Script principal usado
- {script_name}

Funciones principales llamadas
- torch.load
- AutoConfig.from_pretrained
- AutoTokenizer.from_pretrained
- Gazesup_BERTForCombinedMaskedLM.from_pretrained
- BertModel.save_pretrained
- AutoModelForSequenceClassification.from_pretrained

Output generado
- Pasos/paso_8/salida_export_backbone_debug.txt con la inspeccion y verificacion de export.
- Pasos/paso_8/README_paso_8.txt con la explicacion del puente tecnico hacia downstream.
- Pasos/paso_8/comandos_y_funciones.txt con trazabilidad de la corrida.
- Pasos/paso_8/{args.export_dirname}/ con el backbone exportado en formato reutilizable.
"""


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = inspect_and_export(args)
    script_name = Path(__file__).name

    (output_dir / "salida_export_backbone_debug.txt").write_text(build_debug_output(summary), encoding="utf-8")
    (output_dir / "README_paso_8.txt").write_text(build_readme(args, summary, script_name), encoding="utf-8")
    (output_dir / "comandos_y_funciones.txt").write_text(build_commands_file(args, script_name), encoding="utf-8")

    print(f"Wrote {output_dir / 'salida_export_backbone_debug.txt'}")
    print(f"Wrote {output_dir / 'README_paso_8.txt'}")
    print(f"Wrote {output_dir / 'comandos_y_funciones.txt'}")
    print(f"Exported backbone to {summary['export_dir']}")


if __name__ == "__main__":
    main()
