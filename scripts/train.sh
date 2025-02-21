SIMPLE="FALSE"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        -s|--simple) SIMPLE="TRUE"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ "$SIMPLE" = "TRUE" ]; then
    python training/finetune_pixtral_12b_simple.py --config configs/pixtral_12b_casters.yaml 
else
    python training/finetune_pixtral_12b.py --config configs/pixtral_12b_casters.yaml 
fi