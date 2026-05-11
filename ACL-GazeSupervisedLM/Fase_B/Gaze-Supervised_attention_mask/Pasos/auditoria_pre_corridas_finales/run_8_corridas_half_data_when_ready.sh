#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_DIR"

POLL_INTERVAL_SECONDS="${POLL_INTERVAL_SECONDS:-60}"
MIN_FREE_DISK_GB="${MIN_FREE_DISK_GB:-25}"
MIN_FREE_DISK_KB=$((MIN_FREE_DISK_GB * 1024 * 1024))
LAUNCHER="Pasos/auditoria_pre_corridas_finales/comandos_8_corridas_half_data.sh"

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

free_disk_kb() {
  df -Pk "$REPO_DIR" | awk 'NR == 2 {print $4}'
}

log "Watcher iniciado para las 8 corridas half-data."
log "Condiciones: nvidia-smi OK y disco libre >= ${MIN_FREE_DISK_GB}G."
log "Launcher: $LAUNCHER"

while true; do
  if ! nvidia-smi -L >/dev/null 2>&1; then
    log "Esperando GPU: nvidia-smi todavia no responde."
    sleep "$POLL_INTERVAL_SECONDS"
    continue
  fi

  current_free_kb="$(free_disk_kb)"
  current_free_gb="$((current_free_kb / 1024 / 1024))"
  if (( current_free_kb < MIN_FREE_DISK_KB )); then
    log "Esperando espacio: libres ${current_free_gb}G, requerido >= ${MIN_FREE_DISK_GB}G."
    sleep "$POLL_INTERVAL_SECONDS"
    continue
  fi

  log "Condiciones OK. Lanzando las 8 corridas y graficos."
  break
done

export RUN_EXPERIMENTS=1
export PYTHONUNBUFFERED=1

# Mitiga el SIGILL visto en libtorch_cpu.so si alguna parte cae a CPU.
export MKL_ENABLE_INSTRUCTIONS="${MKL_ENABLE_INSTRUCTIONS:-AVX2}"
export ATEN_CPU_CAPABILITY="${ATEN_CPU_CAPABILITY:-avx2}"

exec bash "$LAUNCHER"
