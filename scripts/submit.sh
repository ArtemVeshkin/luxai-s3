#!/bin/bash

if [ -z $LUXAI_ROOT_PATH ]
then
    echo "LUXAI_ROOT_PATH is not specified!"
    echo "Run: export LUXAI_ROOT_PATH=<path to project>"
    echo "For example: export LUXAI_ROOT_PATH=/home/artemveshkin/dev/luxai-s3"
else
    if [ -z "$1" ]
    then 
        echo "Pass agent name"
        echo "Example:"
        echo "  sh scripts/submit.sh baseline"
    else
        echo "Creating submit for $1 agent"
        SCRIPT_WORK_DIR="$LUXAI_ROOT_PATH/submit"
        KIT_PATH="Lux-Design-S3/kits/python"

        rm -rf $SCRIPT_WORK_DIR && mkdir -p $SCRIPT_WORK_DIR
        cp -r "$LUXAI_ROOT_PATH/agents/$1/." $SCRIPT_WORK_DIR

        cp "$LUXAI_ROOT_PATH/$KIT_PATH/main.py" $SCRIPT_WORK_DIR
        cp -rp "$LUXAI_ROOT_PATH/$KIT_PATH/lux" $SCRIPT_WORK_DIR

        cd $SCRIPT_WORK_DIR && tar -czvf "$LUXAI_ROOT_PATH/submit.tar.gz" *
        rm -rf $SCRIPT_WORK_DIR
    fi
fi