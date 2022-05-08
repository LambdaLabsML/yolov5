#!/bin/bash

SYSTEM=${1:-"a6000"}
TASK_NAME=${2:-"all"}

declare -A TASKS=(
    [YOLOv5n]=yolov5n
    [YOLOv5s]=yolov5s
    [YOLOv5m]=yolov5m
    [YOLOv5l]=yolov5l
    [YOLOv5x]=yolov5x
    [YOLOv5n6]=yolov5n6
    [YOLOv5s6]=yolov5s6
    [YOLOv5m6]=yolov5m6
    [YOLOv5l6]=yolov5l6
    [YOLOv5x6TTA]=yolov5x6
)

declare -A IMGSZ=(
    [YOLOv5n]=640
    [YOLOv5s]=640
    [YOLOv5m]=640
    [YOLOv5l]=640
    [YOLOv5x]=640
    [YOLOv5n6]=1280
    [YOLOv5s6]=1280
    [YOLOv5m6]=1280
    [YOLOv5l6]=1280
    [YOLOv5x6TTA]=1280
)


main() {
    FOLDER=./lambda/benchmarks/{SYSTEM}-$(date +"%m-%d-%y-%H-%M")
    mkdir -p $FOLDER
    for task in "${!TASKS[@]}"; do
        if [[ "${task,,}" == *"$TASK_NAME"* ]] || [ "$TASK_NAME" == "all" ]; then
            python ./utils/benchmarks.py --weights ${TASKS[${task}]}.pt --img ${IMGSZ[${task}]} --device 0 |& tee $FOLDER/${task}.txt
        fi
    done
}

main "$@"
