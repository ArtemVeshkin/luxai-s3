#!/bin/bash

if [ -z "$2" ]
then 
    echo "Pass agent names"
    echo "Example:"
    echo "  sh scripts/run.sh baseline baseline"
else
    echo "Run $1 agent VS $2 agent"

    BASE_DIR="$HOME/dev/luxai-s3" # CHANGE IT!!!
    SCRIPT_WORK_DIR="$BASE_DIR/runs/$1_vs_$2"
    KIT_PATH="Lux-Design-S3/kits/python"

    AGENT_1_PATH="$SCRIPT_WORK_DIR/agent_1"
    AGENT_2_PATH="$SCRIPT_WORK_DIR/agent_2"

    rm -rf $SCRIPT_WORK_DIR && mkdir -p $SCRIPT_WORK_DIR
    mkdir $AGENT_1_PATH && cp -r "$BASE_DIR/agents/$1/." $AGENT_1_PATH
    mkdir $AGENT_2_PATH && cp -r "$BASE_DIR/agents/$2/." $AGENT_2_PATH

    cp "$BASE_DIR/$KIT_PATH/main.py" $AGENT_1_PATH
    cp "$BASE_DIR/$KIT_PATH/main.py" $AGENT_2_PATH

    cp -rp "$BASE_DIR/$KIT_PATH/lux" $AGENT_1_PATH
    cp -rp "$BASE_DIR/$KIT_PATH/lux" $AGENT_2_PATH

    luxai-s3 "$AGENT_1_PATH/main.py" "$AGENT_2_PATH/main.py" \
        --output="$SCRIPT_WORK_DIR/replay.html"
fi