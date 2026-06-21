#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "$REPO_ROOT"

TARGET="${MUON_BENCH_TARGET:-${1:-local}}"
PROFILE="${MUON_BENCH_PROFILE:-${2:-fullprod-e8}}"
STAMP="$(date -u +%Y%m%d-%H%M%S)"
DEFAULT_ALLOW_BOUNDARY_COLLECTIVES=false
DEFAULT_NS_COMPUTE_DTYPE=input

case "$PROFILE" in
    fullprod-e8)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L2-E8-FULLPRODMUONH-H1H3H5-N1-cw-${STAMP}"
        DEFAULT_KINDS="expert_only_grouped_muonh_optimizer_apply,expert_grouped_muonh_optimizer_apply,full_production_muonh_optimizer_apply"
        DEFAULT_LAYERS=2
        DEFAULT_GROUP_SIZE=2
        DEFAULT_GROUP_AXIS=none
        DEFAULT_REPLICA_AXIS=1
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=1
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=256g
        DEFAULT_STEPS="1,3,5"
        DEFAULT_CAPS="512"
        ;;
    fullprod-e8-h3)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L2-E8-FULLPRODMUONH-H3-N1-cw-${STAMP}"
        DEFAULT_KINDS="expert_only_grouped_muonh_optimizer_apply,expert_grouped_muonh_optimizer_apply,full_production_muonh_optimizer_apply"
        DEFAULT_LAYERS=2
        DEFAULT_GROUP_SIZE=2
        DEFAULT_GROUP_AXIS=none
        DEFAULT_REPLICA_AXIS=1
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=1
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=256g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    fullprod-e8-l26-h3)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-E8-FULLPRODMUONH-G4-H3-N1-cw-${STAMP}"
        DEFAULT_KINDS="expert_only_grouped_muonh_optimizer_apply,expert_grouped_muonh_optimizer_apply,full_production_muonh_optimizer_apply"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=4
        DEFAULT_GROUP_AXIS=none
        DEFAULT_REPLICA_AXIS=1
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=1
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=256g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    fullprod-r4e8-l26-h3)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R4E8-FULLPRODMUONH-G4-H3-N4-cw-${STAMP}"
        DEFAULT_KINDS="expert_only_grouped_muonh_optimizer_apply,expert_grouped_muonh_optimizer_apply,full_production_muonh_optimizer_apply"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=4
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=4
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=4
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=256g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    packed-master-fsdp-layer-n1-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-E8-PACKEDMASTER-FSDPLAYER-N1-cw-${STAMP}"
        DEFAULT_KINDS="expert_packed_master_fsdp_layer_consumer"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=26
        DEFAULT_GROUP_AXIS=none
        DEFAULT_REPLICA_AXIS=1
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=1
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    packed-master-r2e8-l26-h3)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R2E8-PACKEDMASTER-H3-N2-cw-${STAMP}"
        DEFAULT_KINDS="expert_packed_master_muonh_consumer,expert_packed_master_fsdp_layer_consumer"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=2
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=2
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    packed-master-r4e8-l26-h3)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R4E8-PACKEDMASTER-H3-N4-cw-${STAMP}"
        DEFAULT_KINDS="expert_packed_master_muonh_consumer,expert_packed_master_fsdp_layer_consumer"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=4
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=4
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    packed-master-fsdp-seq-r4e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R4E8-PACKEDMASTER-FSDPSEQ-N4-cw-${STAMP}"
        DEFAULT_KINDS="expert_packed_master_fsdp_sequential_consumer"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=4
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=4
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    packed-master-fsdp-slab-r4e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R4E8-PACKEDMASTER-FSDPSLAB-N4-cw-${STAMP}"
        DEFAULT_KINDS="expert_packed_master_fsdp_slab_consumer"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=4
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=4
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    packed-master-fsdp-bulk-r4e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R4E8-PACKEDMASTER-FSDPBULK-N4-cw-${STAMP}"
        DEFAULT_KINDS="expert_packed_master_fsdp_bulk_consumer"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=4
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=4
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    packed-master-fsdp-grad-seq-r4e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R4E8-PACKEDMASTER-FSDPGRADSEQ-N4-cw-${STAMP}"
        DEFAULT_KINDS="expert_packed_master_fsdp_sequential_grad"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=4
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=4
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    packed-master-fsdp-grad-slab-r4e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R4E8-PACKEDMASTER-FSDPGRADSLAB-N4-cw-${STAMP}"
        DEFAULT_KINDS="expert_packed_master_fsdp_slab_grad"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=4
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=4
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    packed-master-fsdp-grad-bulk-r4e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R4E8-PACKEDMASTER-FSDPGRADBULK-N4-cw-${STAMP}"
        DEFAULT_KINDS="expert_packed_master_fsdp_bulk_grad"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=4
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=4
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    packed-master-bulk-grad-muonh-r4e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R4E8-PACKEDMASTER-BULKGRAD-MUONH-N4-cw-${STAMP}"
        DEFAULT_KINDS="expert_packed_master_bulk_grad_muonh_update"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=4
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=4
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    packed-master-bulk-grad-split-muonh-r4e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R4E8-PACKEDMASTER-BULKGRAD-SPLITMUONH-N4-cw-${STAMP}"
        DEFAULT_KINDS="expert_packed_master_bulk_grad_split_muonh_update"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=4
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=4
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    packed-master-muonh-update-only-r4e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R4E8-PACKEDMASTER-MUONH-UPDATEONLY-N4-cw-${STAMP}"
        DEFAULT_KINDS="expert_packed_master_muonh_update_only"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=4
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=4
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    packed-master-muonh-checksum-r4e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R4E8-PACKEDMASTER-MUONH-CHECKSUM-N4-cw-${STAMP}"
        DEFAULT_KINDS="expert_packed_master_muonh_update_checksum"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=4
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=4
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    packed-master-muonh-decomp-r4e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R4E8-PACKEDMASTER-MUONH-DECOMP-N4-cw-${STAMP}"
        DEFAULT_KINDS="expert_packed_master_momentum_checksum,expert_packed_master_momentum_direction,expert_packed_master_ns_checksum,expert_packed_master_ns_hyperball_checksum,expert_packed_master_ns_hyperball_apply_checksum,expert_packed_master_direction_ns_hyperball_checksum,expert_packed_master_muonh_update_checksum"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=4
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=4
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    packed-master-muonh-chunked-r4e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R4E8-PACKEDMASTER-MUONH-CHUNKED-N4-cw-${STAMP}"
        DEFAULT_KINDS="expert_packed_master_chunked_direction_ns_hyperball_checksum"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=4
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=4
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    packed-master-muonh-chunked-update-r4e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R4E8-PACKEDMASTER-MUONH-CHUNKEDUPDATE-N4-cw-${STAMP}"
        DEFAULT_KINDS="expert_packed_master_chunked_muonh_update_only"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=4
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=4
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    chunked-packed-master-muonh-update-r4e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R4E8-CHUNKEDPACKEDMASTER-MUONH-UPDATE-N4-cw-${STAMP}"
        DEFAULT_KINDS="expert_chunked_packed_master_muonh_update_only"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=4
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=4
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    chunked-packed-master-fsdp-layer-r4e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R4E8-CHUNKEDPACKEDMASTER-FSDPLAYER-N4-cw-${STAMP}"
        DEFAULT_KINDS="expert_chunked_packed_master_fsdp_layer_consumer"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=4
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=4
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    chunked-packed-master-fsdp-seq-r4e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R4E8-CHUNKEDPACKEDMASTER-FSDPSEQ-N4-cw-${STAMP}"
        DEFAULT_KINDS="expert_chunked_packed_master_fsdp_sequential_consumer"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=4
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=4
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    chunked-packed-master-sequential-grad-muonh-n1-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-E8-CHUNKEDPACKEDMASTER-SEQUENTIALGRAD-MUONH-N1-cw-${STAMP}"
        DEFAULT_KINDS="expert_chunked_packed_master_fsdp_sequential_grad_muonh_update"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=none
        DEFAULT_REPLICA_AXIS=1
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=1
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    chunked-packed-master-sequential-grad-muonh-checksum-n1-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-E8-CHUNKEDPACKEDMASTER-SEQUENTIALGRAD-MUONH-CHECKSUM-N1-cw-${STAMP}"
        DEFAULT_KINDS="expert_chunked_packed_master_fsdp_sequential_grad_muonh_checksum"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=none
        DEFAULT_REPLICA_AXIS=1
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=1
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    chunked-packed-master-streaming-grad-muonh-checksum-n1-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-E8-CHUNKEDPACKEDMASTER-STREAMINGGRAD-MUONH-CHECKSUM-N1-cw-${STAMP}"
        DEFAULT_KINDS="expert_chunked_packed_master_fsdp_streaming_grad_muonh_checksum"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=none
        DEFAULT_REPLICA_AXIS=1
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=1
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    layer-chunked-packed-master-streaming-grad-muonh-checksum-n1-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-E8-LAYERCHUNKEDPACKEDMASTER-STREAMINGGRAD-MUONH-CHECKSUM-N1-cw-${STAMP}"
        DEFAULT_KINDS="expert_layer_chunked_packed_master_fsdp_streaming_grad_muonh_checksum"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=none
        DEFAULT_REPLICA_AXIS=1
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=1
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    layer-chunked-packed-master-streaming-grad-muonh-update-n1-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-E8-LAYERCHUNKEDPACKEDMASTER-STREAMINGGRAD-MUONH-UPDATE-N1-cw-${STAMP}"
        DEFAULT_KINDS="expert_layer_chunked_packed_master_fsdp_streaming_grad_muonh_update"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=none
        DEFAULT_REPLICA_AXIS=1
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=1
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    layer-chunked-packed-master-streaming-value-grad-muonh-update-n1-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-E8-LAYERCHUNKEDPACKEDMASTER-STREAMINGVALUEGRAD-MUONH-UPDATE-N1-cw-${STAMP}"
        DEFAULT_KINDS="expert_layer_chunked_packed_master_fsdp_streaming_value_grad_muonh_update"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=none
        DEFAULT_REPLICA_AXIS=1
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=1
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        DEFAULT_NS_COMPUTE_DTYPE=bf16
        ;;
    layer-chunked-packed-master-streaming-grad-muonh-next-loss-n1-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-E8-LAYERCHUNKEDPACKEDMASTER-STREAMINGGRAD-MUONH-NEXTLOSS-N1-cw-${STAMP}"
        DEFAULT_KINDS="expert_layer_chunked_packed_master_fsdp_streaming_grad_muonh_next_loss"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=none
        DEFAULT_REPLICA_AXIS=1
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=1
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    chunked-packed-master-sequential-grad-muonh-r2e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R2E8-CHUNKEDPACKEDMASTER-SEQUENTIALGRAD-MUONH-N2-cw-${STAMP}"
        DEFAULT_KINDS="expert_chunked_packed_master_fsdp_sequential_grad_muonh_update"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=2
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=2
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    chunked-packed-master-sequential-grad-muonh-checksum-r2e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R2E8-CHUNKEDPACKEDMASTER-SEQUENTIALGRAD-MUONH-CHECKSUM-N2-cw-${STAMP}"
        DEFAULT_KINDS="expert_chunked_packed_master_fsdp_sequential_grad_muonh_checksum"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=2
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=2
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    chunked-packed-master-streaming-grad-muonh-checksum-r2e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R2E8-CHUNKEDPACKEDMASTER-STREAMINGGRAD-MUONH-CHECKSUM-N2-cw-${STAMP}"
        DEFAULT_KINDS="expert_chunked_packed_master_fsdp_streaming_grad_muonh_checksum"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=2
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=2
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    layer-chunked-packed-master-streaming-grad-muonh-checksum-r2e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R2E8-LAYERCHUNKEDPACKEDMASTER-STREAMINGGRAD-MUONH-CHECKSUM-N2-cw-${STAMP}"
        DEFAULT_KINDS="expert_layer_chunked_packed_master_fsdp_streaming_grad_muonh_checksum"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=2
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=2
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    layer-chunked-packed-master-streaming-grad-muonh-update-r2e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R2E8-LAYERCHUNKEDPACKEDMASTER-STREAMINGGRAD-MUONH-UPDATE-N2-cw-${STAMP}"
        DEFAULT_KINDS="expert_layer_chunked_packed_master_fsdp_streaming_grad_muonh_update"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=2
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=2
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    layer-chunked-packed-master-streaming-value-grad-muonh-update-r2e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R2E8-LAYERCHUNKEDPACKEDMASTER-STREAMINGVALUEGRAD-MUONH-UPDATE-N2-cw-${STAMP}"
        DEFAULT_KINDS="expert_layer_chunked_packed_master_fsdp_streaming_value_grad_muonh_update"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=2
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=2
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        DEFAULT_NS_COMPUTE_DTYPE=bf16
        ;;
    layer-chunked-packed-master-streaming-layerwise-value-grad-muonh-update-r2e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R2E8-LAYERCHUNKEDPACKEDMASTER-STREAMINGLAYERWISEVALUEGRAD-MUONH-UPDATE-N2-cw-${STAMP}"
        DEFAULT_KINDS="expert_layer_chunked_packed_master_fsdp_streaming_layerwise_value_grad_muonh_update"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=2
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=2
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    layer-chunked-packed-master-streaming-block-group-value-grad-muonh-update-r2e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R2E8-LAYERCHUNKEDPACKEDMASTER-STREAMINGBLOCKGROUPVALUEGRAD-MUONH-UPDATE-N2-cw-${STAMP}"
        DEFAULT_KINDS="expert_layer_chunked_packed_master_fsdp_streaming_block_group_value_grad_muonh_update"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=2
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=2
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        DEFAULT_NS_COMPUTE_DTYPE=bf16
        ;;
    layer-chunked-packed-master-streaming-grad-muonh-next-loss-r2e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R2E8-LAYERCHUNKEDPACKEDMASTER-STREAMINGGRAD-MUONH-NEXTLOSS-N2-cw-${STAMP}"
        DEFAULT_KINDS="expert_layer_chunked_packed_master_fsdp_streaming_grad_muonh_next_loss"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=2
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=2
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    layer-chunked-packed-master-grouped-grad-muonh-update-r2e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R2E8-LAYERCHUNKEDPACKEDMASTER-GROUPEDGRAD-MUONH-UPDATE-N2-cw-${STAMP}"
        DEFAULT_KINDS="expert_layer_chunked_packed_master_grouped_grad_muonh_update"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=2
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=2
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    chunked-packed-master-sequential-grad-muonh-r4e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R4E8-CHUNKEDPACKEDMASTER-SEQUENTIALGRAD-MUONH-N4-cw-${STAMP}"
        DEFAULT_KINDS="expert_chunked_packed_master_fsdp_sequential_grad_muonh_update"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=4
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=4
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    chunked-packed-master-streaming-grad-muonh-checksum-r4e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R4E8-CHUNKEDPACKEDMASTER-STREAMINGGRAD-MUONH-CHECKSUM-N4-cw-${STAMP}"
        DEFAULT_KINDS="expert_chunked_packed_master_fsdp_streaming_grad_muonh_checksum"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=4
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=4
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    layer-chunked-packed-master-streaming-grad-muonh-checksum-r4e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R4E8-LAYERCHUNKEDPACKEDMASTER-STREAMINGGRAD-MUONH-CHECKSUM-N4-cw-${STAMP}"
        DEFAULT_KINDS="expert_layer_chunked_packed_master_fsdp_streaming_grad_muonh_checksum"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=4
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=4
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    layer-chunked-packed-master-streaming-grad-muonh-update-r4e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R4E8-LAYERCHUNKEDPACKEDMASTER-STREAMINGGRAD-MUONH-UPDATE-N4-cw-${STAMP}"
        DEFAULT_KINDS="expert_layer_chunked_packed_master_fsdp_streaming_grad_muonh_update"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=4
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=4
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    layer-chunked-packed-master-streaming-value-grad-muonh-update-r4e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R4E8-LAYERCHUNKEDPACKEDMASTER-STREAMINGVALUEGRAD-MUONH-UPDATE-N4-cw-${STAMP}"
        DEFAULT_KINDS="expert_layer_chunked_packed_master_fsdp_streaming_value_grad_muonh_update"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=4
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=4
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        DEFAULT_NS_COMPUTE_DTYPE=bf16
        ;;
    layer-chunked-packed-master-streaming-layerwise-value-grad-muonh-update-r4e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R4E8-LAYERCHUNKEDPACKEDMASTER-STREAMINGLAYERWISEVALUEGRAD-MUONH-UPDATE-N4-cw-${STAMP}"
        DEFAULT_KINDS="expert_layer_chunked_packed_master_fsdp_streaming_layerwise_value_grad_muonh_update"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=4
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=4
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        DEFAULT_NS_COMPUTE_DTYPE=bf16
        ;;
    layer-chunked-packed-master-streaming-block-group-value-grad-muonh-update-r4e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R4E8-LAYERCHUNKEDPACKEDMASTER-STREAMINGBLOCKGROUPVALUEGRAD-MUONH-UPDATE-N4-cw-${STAMP}"
        DEFAULT_KINDS="expert_layer_chunked_packed_master_fsdp_streaming_block_group_value_grad_muonh_update"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=4
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=4
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        DEFAULT_NS_COMPUTE_DTYPE=bf16
        ;;
    layer-chunked-packed-master-fsdp-seq-r4e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R4E8-LAYERCHUNKEDPACKEDMASTER-FSDPSEQ-N4-cw-${STAMP}"
        DEFAULT_KINDS="expert_layer_chunked_packed_master_fsdp_sequential_consumer"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=4
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=4
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    layer-chunked-packed-master-streaming-grad-muonh-next-loss-r4e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R4E8-LAYERCHUNKEDPACKEDMASTER-STREAMINGGRAD-MUONH-NEXTLOSS-N4-cw-${STAMP}"
        DEFAULT_KINDS="expert_layer_chunked_packed_master_fsdp_streaming_grad_muonh_next_loss"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=4
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=4
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    layer-chunked-packed-master-grouped-grad-muonh-update-r4e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R4E8-LAYERCHUNKEDPACKEDMASTER-GROUPEDGRAD-MUONH-UPDATE-N4-cw-${STAMP}"
        DEFAULT_KINDS="expert_layer_chunked_packed_master_grouped_grad_muonh_update"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=4
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=4
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    chunked-packed-master-sequential-grad-muonh-checksum-r4e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R4E8-CHUNKEDPACKEDMASTER-SEQUENTIALGRAD-MUONH-CHECKSUM-N4-cw-${STAMP}"
        DEFAULT_KINDS="expert_chunked_packed_master_fsdp_sequential_grad_muonh_checksum"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=4
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=4
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    packed-master-muonh-fsdp-seq-r4e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R4E8-PACKEDMASTER-MUONH-FSDPSEQ-N4-cw-${STAMP}"
        DEFAULT_KINDS="expert_packed_master_muonh_fsdp_sequential_consumer"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=4
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=4
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    packed-master-muonh-fsdp-slab-r4e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R4E8-PACKEDMASTER-MUONH-FSDPSLAB-N4-cw-${STAMP}"
        DEFAULT_KINDS="expert_packed_master_muonh_fsdp_slab_consumer"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=4
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=4
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    packed-master-muonh-fsdp-bulk-r4e8-l26)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R4E8-PACKEDMASTER-MUONH-FSDPBULK-N4-cw-${STAMP}"
        DEFAULT_KINDS="expert_packed_master_muonh_fsdp_bulk_consumer"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=8
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=4
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=4
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=512g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    grouped2d-decomp-r4e8-l26-h3)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R4E8-GROUPED2D-DECOMP-G4-H3-N4-cw-${STAMP}"
        DEFAULT_KINDS="expert_only_grouped_muonh_optimizer_apply,expert_grouped_muonh_optimizer_apply,full_production_muonh_optimizer_apply,full_production_grouped_2d_muonh_optimizer_apply,ordinary_2d_muonh_optimizer_apply,ordinary_2d_grouped_muonh_optimizer_apply,ordinary_2d_grouped_stack_ns,ordinary_2d_grouped_restore_split,full_production_apply_only"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=4
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=4
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=4
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=256g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        DEFAULT_ALLOW_BOUNDARY_COLLECTIVES=true
        ;;
    fullprod-r16e8-l26-h3)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R16E8-FULLPRODMUONH-G16-H3-N16-cw-${STAMP}"
        DEFAULT_KINDS="expert_only_grouped_muonh_optimizer_apply,expert_grouped_muonh_optimizer_apply,full_production_muonh_optimizer_apply"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=16
        DEFAULT_GROUP_AXIS=replica_dcn
        DEFAULT_REPLICA_AXIS=16
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=16
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=256g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    grouped-d2e4)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L2-D2E4-GROUPEDMUONH-H1H3H5-N1-cw-${STAMP}"
        DEFAULT_KINDS="expert_only_grouped_muonh_optimizer_apply,expert_grouped_muonh_optimizer_apply,ns4d_data_group_apply,ns4d_data_reshard_restore"
        DEFAULT_LAYERS=2
        DEFAULT_GROUP_SIZE=2
        DEFAULT_GROUP_AXIS=data
        DEFAULT_REPLICA_AXIS=1
        DEFAULT_DATA_AXIS=2
        DEFAULT_EXPERT_AXIS=4
        DEFAULT_GPU_REPLICAS=1
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=256g
        DEFAULT_STEPS="1,3,5"
        DEFAULT_CAPS="512"
        ;;
    expert-only-e8-l26-h3)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-E8-EXPERTONLY-H3-N1-cw-${STAMP}"
        DEFAULT_KINDS="expert_only_grouped_muonh_optimizer_apply"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=26
        DEFAULT_GROUP_AXIS=none
        DEFAULT_REPLICA_AXIS=1
        DEFAULT_DATA_AXIS=1
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=1
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=256g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    expert-only-r2d2e8-l26-h3)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R2D2E8-EXPERTONLY-H3-N4-cw-${STAMP}"
        DEFAULT_KINDS="expert_only_grouped_muonh_optimizer_apply"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=26
        DEFAULT_GROUP_AXIS=replica_dcn,data
        DEFAULT_REPLICA_AXIS=2
        DEFAULT_DATA_AXIS=2
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=4
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=256g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        ;;
    expert-fsdp-r2d2e8-l26-h3)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L26-R2D2E8-EXPERTFSDP-H3-N4-cw-${STAMP}"
        DEFAULT_KINDS="expert_only_grouped_muonh_optimizer_apply,expert_fsdp_grouped_updates_muonh_apply,expert_fsdp_grouped_updates_muonh_explicit_apply,expert_fsdp_grouped_updates_muonh_explicit_a2a_apply"
        DEFAULT_LAYERS=26
        DEFAULT_GROUP_SIZE=26
        DEFAULT_GROUP_AXIS=replica_dcn,data
        DEFAULT_REPLICA_AXIS=2
        DEFAULT_DATA_AXIS=2
        DEFAULT_EXPERT_AXIS=8
        DEFAULT_GPU_REPLICAS=4
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=256g
        DEFAULT_STEPS="3"
        DEFAULT_CAPS="512"
        DEFAULT_ALLOW_BOUNDARY_COLLECTIVES=true
        ;;
    padding-d2e4)
        DEFAULT_RUN_ID="MUON-BENCH-D2560-L2-D2E4-PADDING-H1H3H5-N1-cw-${STAMP}"
        DEFAULT_KINDS="ns4d_data_group,ns4d_padded_group,ns4d_dotonly_matmul_padded"
        DEFAULT_LAYERS=2
        DEFAULT_GROUP_SIZE=3
        DEFAULT_GROUP_AXIS=data
        DEFAULT_REPLICA_AXIS=1
        DEFAULT_DATA_AXIS=2
        DEFAULT_EXPERT_AXIS=4
        DEFAULT_GPU_REPLICAS=1
        DEFAULT_WORKER_CPU=8
        DEFAULT_WORKER_RAM=256g
        DEFAULT_STEPS="1,3,5"
        DEFAULT_CAPS="512"
        ;;
    *)
        echo "Unknown MUON_BENCH_PROFILE=${PROFILE}" >&2
        echo "Profiles: fullprod-e8, fullprod-e8-h3, fullprod-e8-l26-h3, fullprod-r4e8-l26-h3, packed-master-fsdp-layer-n1-l26, packed-master-r2e8-l26-h3, packed-master-r4e8-l26-h3, packed-master-fsdp-seq-r4e8-l26, packed-master-fsdp-slab-r4e8-l26, packed-master-fsdp-bulk-r4e8-l26, packed-master-fsdp-grad-seq-r4e8-l26, packed-master-fsdp-grad-slab-r4e8-l26, packed-master-fsdp-grad-bulk-r4e8-l26, packed-master-bulk-grad-muonh-r4e8-l26, packed-master-bulk-grad-split-muonh-r4e8-l26, packed-master-muonh-update-only-r4e8-l26, packed-master-muonh-checksum-r4e8-l26, packed-master-muonh-decomp-r4e8-l26, packed-master-muonh-chunked-r4e8-l26, packed-master-muonh-chunked-update-r4e8-l26, chunked-packed-master-muonh-update-r4e8-l26, chunked-packed-master-sequential-grad-muonh-n1-l26, chunked-packed-master-sequential-grad-muonh-r2e8-l26, chunked-packed-master-sequential-grad-muonh-r4e8-l26, chunked-packed-master-sequential-grad-muonh-checksum-n1-l26, chunked-packed-master-sequential-grad-muonh-checksum-r2e8-l26, chunked-packed-master-sequential-grad-muonh-checksum-r4e8-l26, chunked-packed-master-streaming-grad-muonh-checksum-n1-l26, layer-chunked-packed-master-streaming-grad-muonh-checksum-n1-l26, layer-chunked-packed-master-streaming-grad-muonh-update-n1-l26, layer-chunked-packed-master-streaming-value-grad-muonh-update-n1-l26, layer-chunked-packed-master-streaming-grad-muonh-next-loss-n1-l26, chunked-packed-master-streaming-grad-muonh-checksum-r2e8-l26, layer-chunked-packed-master-streaming-grad-muonh-update-r2e8-l26, layer-chunked-packed-master-streaming-value-grad-muonh-update-r2e8-l26, layer-chunked-packed-master-streaming-block-group-value-grad-muonh-update-r2e8-l26, layer-chunked-packed-master-streaming-grad-muonh-next-loss-r2e8-l26, layer-chunked-packed-master-grouped-grad-muonh-update-r2e8-l26, chunked-packed-master-streaming-grad-muonh-checksum-r4e8-l26, layer-chunked-packed-master-streaming-grad-muonh-update-r4e8-l26, layer-chunked-packed-master-streaming-value-grad-muonh-update-r4e8-l26, layer-chunked-packed-master-streaming-block-group-value-grad-muonh-update-r4e8-l26, layer-chunked-packed-master-fsdp-seq-r4e8-l26, layer-chunked-packed-master-streaming-grad-muonh-next-loss-r4e8-l26, layer-chunked-packed-master-grouped-grad-muonh-update-r4e8-l26, packed-master-muonh-fsdp-seq-r4e8-l26, packed-master-muonh-fsdp-slab-r4e8-l26, packed-master-muonh-fsdp-bulk-r4e8-l26, grouped2d-decomp-r4e8-l26-h3, fullprod-r16e8-l26-h3, grouped-d2e4, expert-only-e8-l26-h3, expert-only-r2d2e8-l26-h3, expert-fsdp-r2d2e8-l26-h3, padding-d2e4" >&2
        exit 2
        ;;
esac

RUN_ID="${RUN_ID:-$DEFAULT_RUN_ID}"
MUON_BENCH_LAYERS="${MUON_BENCH_LAYERS:-$DEFAULT_LAYERS}"
MUON_BENCH_NS4D_GROUP_SIZE="${MUON_BENCH_NS4D_GROUP_SIZE:-$DEFAULT_GROUP_SIZE}"
MUON_BENCH_NS4D_GROUP_AXIS="${MUON_BENCH_NS4D_GROUP_AXIS:-$DEFAULT_GROUP_AXIS}"
MUON_BENCH_REPLICA_AXIS="${MUON_BENCH_REPLICA_AXIS:-$DEFAULT_REPLICA_AXIS}"
MUON_BENCH_DATA_AXIS="${MUON_BENCH_DATA_AXIS:-$DEFAULT_DATA_AXIS}"
MUON_BENCH_EXPERT_AXIS="${MUON_BENCH_EXPERT_AXIS:-$DEFAULT_EXPERT_AXIS}"
MUON_BENCH_MODEL_AXIS="${MUON_BENCH_MODEL_AXIS:-1}"
MUON_BENCH_GPU_REPLICAS="${MUON_BENCH_GPU_REPLICAS:-$DEFAULT_GPU_REPLICAS}"
MUON_BENCH_HIDDEN_DIM="${MUON_BENCH_HIDDEN_DIM:-2560}"
MUON_BENCH_INTERMEDIATE_DIM="${MUON_BENCH_INTERMEDIATE_DIM:-1280}"
MUON_BENCH_NUM_EXPERTS="${MUON_BENCH_NUM_EXPERTS:-256}"
MUON_BENCH_DTYPE="${MUON_BENCH_DTYPE:-bf16}"
MUON_BENCH_NS_COMPUTE_DTYPE="${MUON_BENCH_NS_COMPUTE_DTYPE:-$DEFAULT_NS_COMPUTE_DTYPE}"
MUON_BENCH_NESTEROV="${MUON_BENCH_NESTEROV:-true}"
MUON_BENCH_EXPERT_GROUPED_MUONH_PACKED_ENTRY="${MUON_BENCH_EXPERT_GROUPED_MUONH_PACKED_ENTRY:-false}"
MUON_BENCH_EXPERT_GROUPED_MUONH_PACKED_BANK_COMPUTE="${MUON_BENCH_EXPERT_GROUPED_MUONH_PACKED_BANK_COMPUTE:-false}"
MUON_BENCH_EXPERT_GROUPED_MUONH_CHUNK_LOCAL_BOUNDARIES="${MUON_BENCH_EXPERT_GROUPED_MUONH_CHUNK_LOCAL_BOUNDARIES:-false}"
MUON_BENCH_UNFUSED_EXPERT_GATE_UP="${MUON_BENCH_UNFUSED_EXPERT_GATE_UP:-false}"
MUON_BENCH_GROUPED_EXPERT_CONSUMER_TOKENS_PER_EXPERT="${MUON_BENCH_GROUPED_EXPERT_CONSUMER_TOKENS_PER_EXPERT:-1}"
MUON_BENCH_GROUPED_EXPERT_CONSUMER_CHUNK_TOKENS="${MUON_BENCH_GROUPED_EXPERT_CONSUMER_CHUNK_TOKENS:-0}"
MUON_BENCH_GROUPED_EXPERT_CONSUMER_CHUNK_TOKENS_PER_EXPERT="${MUON_BENCH_GROUPED_EXPERT_CONSUMER_CHUNK_TOKENS_PER_EXPERT:-0}"
MUON_BENCH_PACKED_MASTER_LAYER_CHUNK_SIZE="${MUON_BENCH_PACKED_MASTER_LAYER_CHUNK_SIZE:-0}"
MUON_BENCH_ORTHOGONALIZATION_LAYOUT="${MUON_BENCH_ORTHOGONALIZATION_LAYOUT:-stack_batch_4d_sharded}"
MUON_BENCH_SWEEP_BACKEND_STEPS="${MUON_BENCH_SWEEP_BACKEND_STEPS:-$DEFAULT_STEPS}"
MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES="${MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES:-$DEFAULT_CAPS}"
MUON_BENCH_KINDS="${MUON_BENCH_KINDS:-$DEFAULT_KINDS}"
MUON_BENCH_WARMUP="${MUON_BENCH_WARMUP:-1}"
MUON_BENCH_ITERS="${MUON_BENCH_ITERS:-3}"
MUON_BENCH_MODE="${MUON_BENCH_MODE:-both}"
MUON_BENCH_COMPILE_ONLY="${MUON_BENCH_COMPILE_ONLY:-false}"
MUON_BENCH_TRACKER="${MUON_BENCH_TRACKER:-json}"
MUON_BENCH_WANDB="${MUON_BENCH_WANDB:-false}"
MUON_BENCH_WRITE_COMPILED_HLO="${MUON_BENCH_WRITE_COMPILED_HLO:-false}"
MUON_BENCH_ENABLE_JAX_PROFILE="${MUON_BENCH_ENABLE_JAX_PROFILE:-false}"
MUON_BENCH_DISABLE_ABSTRACT_MESH="${MUON_BENCH_DISABLE_ABSTRACT_MESH:-true}"
MUON_BENCH_ALLOW_BOUNDARY_COLLECTIVES="${MUON_BENCH_ALLOW_BOUNDARY_COLLECTIVES:-$DEFAULT_ALLOW_BOUNDARY_COLLECTIVES}"
MUON_BENCH_REQUIRE_NO_BOUNDARY_COLLECTIVES="${MUON_BENCH_REQUIRE_NO_BOUNDARY_COLLECTIVES:-false}"
MUON_BENCH_BOUNDARY_CORRECTNESS_MAX_GLOBAL_BYTES="${MUON_BENCH_BOUNDARY_CORRECTNESS_MAX_GLOBAL_BYTES:-1073741824}"
MUON_BENCH_FORCE_BOUNDARY_CORRECTNESS="${MUON_BENCH_FORCE_BOUNDARY_CORRECTNESS:-false}"
MUON_BENCH_WORKER_CPU="${MUON_BENCH_WORKER_CPU:-$DEFAULT_WORKER_CPU}"
MUON_BENCH_WORKER_RAM="${MUON_BENCH_WORKER_RAM:-$DEFAULT_WORKER_RAM}"
MUON_BENCH_TASK_IMAGE="${MUON_BENCH_TASK_IMAGE:-}"

case "$TARGET" in
    local)
        OUTPUT="${MUON_BENCH_OUTPUT:-scratch/${RUN_ID}.json}"
        LOCAL_DEVICE_COUNT=$((MUON_BENCH_REPLICA_AXIS * MUON_BENCH_DATA_AXIS * MUON_BENCH_EXPERT_AXIS * MUON_BENCH_MODEL_AXIS))
        LOCAL_XLA_FLAGS="${XLA_FLAGS:---xla_force_host_platform_device_count=${LOCAL_DEVICE_COUNT}}"
        DISABLE_ABSTRACT_MESH_ARGS=()
        if [[ "$MUON_BENCH_DISABLE_ABSTRACT_MESH" == "1" || "$MUON_BENCH_DISABLE_ABSTRACT_MESH" == "true" ]]; then
            DISABLE_ABSTRACT_MESH_ARGS=(--disable-abstract-mesh)
        fi
        ALLOW_BOUNDARY_COLLECTIVES_ARGS=()
        if [[ "$MUON_BENCH_ALLOW_BOUNDARY_COLLECTIVES" == "1" || "$MUON_BENCH_ALLOW_BOUNDARY_COLLECTIVES" == "true" ]]; then
            ALLOW_BOUNDARY_COLLECTIVES_ARGS=(--allow-boundary-collectives)
        fi
        FORCE_BOUNDARY_CORRECTNESS_ARGS=()
        if [[ "$MUON_BENCH_FORCE_BOUNDARY_CORRECTNESS" == "1" || "$MUON_BENCH_FORCE_BOUNDARY_CORRECTNESS" == "true" ]]; then
            FORCE_BOUNDARY_CORRECTNESS_ARGS=(--force-boundary-correctness)
        fi
        COMPILE_ONLY_ARGS=()
        if [[ "$MUON_BENCH_COMPILE_ONLY" == "1" || "$MUON_BENCH_COMPILE_ONLY" == "true" ]]; then
            COMPILE_ONLY_ARGS=(--compile-only)
        fi
        NESTEROV_ARGS=()
        if [[ "$MUON_BENCH_NESTEROV" == "0" || "$MUON_BENCH_NESTEROV" == "false" ]]; then
            NESTEROV_ARGS=(--no-nesterov)
        fi
        EXPERT_GROUPED_MUONH_PACKED_ENTRY_ARGS=()
        if [[ "$MUON_BENCH_EXPERT_GROUPED_MUONH_PACKED_ENTRY" == "1" || "$MUON_BENCH_EXPERT_GROUPED_MUONH_PACKED_ENTRY" == "true" ]]; then
            EXPERT_GROUPED_MUONH_PACKED_ENTRY_ARGS=(--expert-grouped-muonh-packed-entry)
        fi
        EXPERT_GROUPED_MUONH_PACKED_BANK_COMPUTE_ARGS=()
        if [[ "$MUON_BENCH_EXPERT_GROUPED_MUONH_PACKED_BANK_COMPUTE" == "1" || "$MUON_BENCH_EXPERT_GROUPED_MUONH_PACKED_BANK_COMPUTE" == "true" ]]; then
            EXPERT_GROUPED_MUONH_PACKED_BANK_COMPUTE_ARGS=(--expert-grouped-muonh-packed-bank-compute)
        fi
        EXPERT_GROUPED_MUONH_CHUNK_LOCAL_BOUNDARIES_ARGS=()
        if [[ "$MUON_BENCH_EXPERT_GROUPED_MUONH_CHUNK_LOCAL_BOUNDARIES" == "1" || "$MUON_BENCH_EXPERT_GROUPED_MUONH_CHUNK_LOCAL_BOUNDARIES" == "true" ]]; then
            EXPERT_GROUPED_MUONH_CHUNK_LOCAL_BOUNDARIES_ARGS=(--expert-grouped-muonh-chunk-local-boundaries)
        fi
        UNFUSED_EXPERT_GATE_UP_ARGS=()
        if [[ "$MUON_BENCH_UNFUSED_EXPERT_GATE_UP" == "1" || "$MUON_BENCH_UNFUSED_EXPERT_GATE_UP" == "true" ]]; then
            UNFUSED_EXPERT_GATE_UP_ARGS=(--unfused-expert-gate-up)
        fi
        PROFILE_ARGS=()
        if [[ "$MUON_BENCH_ENABLE_JAX_PROFILE" == "1" || "$MUON_BENCH_ENABLE_JAX_PROFILE" == "true" ]]; then
            PROFILE_ARGS=(--profile-dir "scratch/profiles/${RUN_ID}")
        fi
        set +u
        XLA_FLAGS="$LOCAL_XLA_FLAGS" \
            uv run python experiments/grug/moe/muon_update_bench.py \
                --layers "$MUON_BENCH_LAYERS" \
                --ns4d-group-size "$MUON_BENCH_NS4D_GROUP_SIZE" \
                --ns4d-group-axis "$MUON_BENCH_NS4D_GROUP_AXIS" \
                --hidden-dim "$MUON_BENCH_HIDDEN_DIM" \
                --intermediate-dim "$MUON_BENCH_INTERMEDIATE_DIM" \
                --num-experts "$MUON_BENCH_NUM_EXPERTS" \
                --dtype "$MUON_BENCH_DTYPE" \
                --ns-compute-dtype "$MUON_BENCH_NS_COMPUTE_DTYPE" \
                --grouped-expert-consumer-tokens-per-expert "$MUON_BENCH_GROUPED_EXPERT_CONSUMER_TOKENS_PER_EXPERT" \
                --grouped-expert-consumer-chunk-tokens "$MUON_BENCH_GROUPED_EXPERT_CONSUMER_CHUNK_TOKENS" \
                --grouped-expert-consumer-chunk-tokens-per-expert "$MUON_BENCH_GROUPED_EXPERT_CONSUMER_CHUNK_TOKENS_PER_EXPERT" \
                --packed-master-layer-chunk-size "$MUON_BENCH_PACKED_MASTER_LAYER_CHUNK_SIZE" \
                --sweep-backend-steps "$MUON_BENCH_SWEEP_BACKEND_STEPS" \
                --sweep-max-grouped-stack-sizes "$MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES" \
                --replica-axis "$MUON_BENCH_REPLICA_AXIS" \
                --data-axis "$MUON_BENCH_DATA_AXIS" \
                --expert-axis "$MUON_BENCH_EXPERT_AXIS" \
                --model-axis "$MUON_BENCH_MODEL_AXIS" \
                --orthogonalization-layout "$MUON_BENCH_ORTHOGONALIZATION_LAYOUT" \
                --bench-kinds "$MUON_BENCH_KINDS" \
                --mode "$MUON_BENCH_MODE" \
                --warmup "$MUON_BENCH_WARMUP" \
                --iters "$MUON_BENCH_ITERS" \
                --boundary-correctness-max-global-bytes "$MUON_BENCH_BOUNDARY_CORRECTNESS_MAX_GLOBAL_BYTES" \
                --output "$OUTPUT" \
                "${DISABLE_ABSTRACT_MESH_ARGS[@]}" \
                "${ALLOW_BOUNDARY_COLLECTIVES_ARGS[@]}" \
                "${FORCE_BOUNDARY_CORRECTNESS_ARGS[@]}" \
                "${COMPILE_ONLY_ARGS[@]}" \
                "${NESTEROV_ARGS[@]}" \
                "${EXPERT_GROUPED_MUONH_PACKED_ENTRY_ARGS[@]}" \
                "${EXPERT_GROUPED_MUONH_PACKED_BANK_COMPUTE_ARGS[@]}" \
                "${EXPERT_GROUPED_MUONH_CHUNK_LOCAL_BOUNDARIES_ARGS[@]}" \
                "${UNFUSED_EXPERT_GATE_UP_ARGS[@]}" \
                "${PROFILE_ARGS[@]}"
        set -u
        echo "Wrote ${OUTPUT}"
        ;;
    iris)
        export RUN_ID
        export MARIN_PREFIX="${MARIN_PREFIX:-s3://marin-na/tmp/ttl=7d}"
        export MUON_BENCH_LAYERS
        export MUON_BENCH_NS4D_GROUP_SIZE
        export MUON_BENCH_NS4D_GROUP_AXIS
        export MUON_BENCH_REPLICA_AXIS
        export MUON_BENCH_DATA_AXIS
        export MUON_BENCH_EXPERT_AXIS
        export MUON_BENCH_MODEL_AXIS
        export MUON_BENCH_HIDDEN_DIM
        export MUON_BENCH_INTERMEDIATE_DIM
        export MUON_BENCH_NUM_EXPERTS
        export MUON_BENCH_DTYPE
        export MUON_BENCH_NS_COMPUTE_DTYPE
        export MUON_BENCH_NESTEROV
        export MUON_BENCH_EXPERT_GROUPED_MUONH_PACKED_ENTRY
        export MUON_BENCH_EXPERT_GROUPED_MUONH_PACKED_BANK_COMPUTE
        export MUON_BENCH_EXPERT_GROUPED_MUONH_CHUNK_LOCAL_BOUNDARIES
        export MUON_BENCH_UNFUSED_EXPERT_GATE_UP
        export MUON_BENCH_GROUPED_EXPERT_CONSUMER_TOKENS_PER_EXPERT
        export MUON_BENCH_GROUPED_EXPERT_CONSUMER_CHUNK_TOKENS
        export MUON_BENCH_GROUPED_EXPERT_CONSUMER_CHUNK_TOKENS_PER_EXPERT
        export MUON_BENCH_PACKED_MASTER_LAYER_CHUNK_SIZE
        export MUON_BENCH_ORTHOGONALIZATION_LAYOUT
        export MUON_BENCH_SWEEP_BACKEND_STEPS
        export MUON_BENCH_SWEEP_MAX_GROUPED_STACK_SIZES
        export MUON_BENCH_KINDS
        export MUON_BENCH_WARMUP
        export MUON_BENCH_ITERS
        export MUON_BENCH_MODE
        export MUON_BENCH_COMPILE_ONLY
        export MUON_BENCH_TRACKER
        export MUON_BENCH_WANDB
        export MUON_BENCH_WRITE_COMPILED_HLO
        export MUON_BENCH_ENABLE_JAX_PROFILE
        export MUON_BENCH_DISABLE_ABSTRACT_MESH
        export MUON_BENCH_ALLOW_BOUNDARY_COLLECTIVES
        export MUON_BENCH_REQUIRE_NO_BOUNDARY_COLLECTIVES
        export MUON_BENCH_BOUNDARY_CORRECTNESS_MAX_GLOBAL_BYTES
        export MUON_BENCH_FORCE_BOUNDARY_CORRECTNESS
        export MUON_BENCH_GPU_REPLICAS
        export MUON_BENCH_WORKER_CPU
        export MUON_BENCH_WORKER_RAM
        export MUON_BENCH_TASK_IMAGE
        export XLA_FLAGS
        export XLA_PYTHON_CLIENT_MEM_FRACTION
        export XLA_PYTHON_CLIENT_ALLOCATOR
        export TF_GPU_ALLOCATOR
        exec bash scratch/launch_muon_update_bench_executor_n1.sh
        ;;
    *)
        echo "Unknown MUON_BENCH_TARGET=${TARGET}; expected local or iris." >&2
        exit 2
        ;;
esac
