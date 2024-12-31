#!/bin/bash


if [ -z $LUXAI_ROOT_PATH ]
then
    echo "LUXAI_ROOT_PATH is not specified!"
    echo "Run: export LUXAI_ROOT_PATH=<path to project>"
    echo "For example: export LUXAI_ROOT_PATH=/home/artemveshkin/dev/luxai-s3"
else
    if [ -z "$2" ]
    then 
        echo "Pass agent names"
        echo "Example:"
        echo "  sh scripts/run.sh baseline baseline"
    else
        echo "Run $1 agent VS $2 agent"

        SCRIPT_WORK_DIR="$LUXAI_ROOT_PATH/runs/$1_vs_$2"
        KIT_PATH="Lux-Design-S3/kits/python"

        AGENT_1_PATH="$SCRIPT_WORK_DIR/$1"
        AGENT_2_PATH="$SCRIPT_WORK_DIR/$2"

        rm -rf $SCRIPT_WORK_DIR && mkdir -p $SCRIPT_WORK_DIR
        mkdir $AGENT_1_PATH && cp -r "$LUXAI_ROOT_PATH/agents/$1/." $AGENT_1_PATH
        mkdir $AGENT_2_PATH && cp -r "$LUXAI_ROOT_PATH/agents/$2/." $AGENT_2_PATH

        cp "$LUXAI_ROOT_PATH/$KIT_PATH/main.py" $AGENT_1_PATH
        cp "$LUXAI_ROOT_PATH/$KIT_PATH/main.py" $AGENT_2_PATH

        cp -rp "$LUXAI_ROOT_PATH/$KIT_PATH/lux" $AGENT_1_PATH
        cp -rp "$LUXAI_ROOT_PATH/$KIT_PATH/lux" $AGENT_2_PATH

        luxai-s3 "$AGENT_1_PATH/main.py" "$AGENT_2_PATH/main.py" \
            --output="$SCRIPT_WORK_DIR/replay.html" --seed=1
    fi
fi