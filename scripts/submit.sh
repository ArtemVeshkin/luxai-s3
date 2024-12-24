#!/bin/bash

if [ -z "$1" ]
then 
    echo "Pass agent name"
    echo "Example:"
    echo "  sh scripts/submit.sh baseline"
else
    echo "Creating submit for $1 agent"

    BASE_DIR="$HOME/dev/luxai-s3" # CHANGE IT!!!
    SCRIPT_WORK_DIR="$BASE_DIR/submit"
    KIT_PATH="Lux-Design-S3/kits/python"

    rm -rf $SCRIPT_WORK_DIR && mkdir -p $SCRIPT_WORK_DIR
    cp -r "$BASE_DIR/agents/$1/." $SCRIPT_WORK_DIR

    cp "$BASE_DIR/$KIT_PATH/main.py" $SCRIPT_WORK_DIR
    cp -rp "$BASE_DIR/$KIT_PATH/lux" $SCRIPT_WORK_DIR

    cd $SCRIPT_WORK_DIR && tar -czvf "$BASE_DIR/submit.tar.gz" *
    rm -rf $SCRIPT_WORK_DIR
fi