#!/bin/bash

echo "Preparing data for intermediate pre-training with STEP..."

python -m STEP.data_gen.create_ud_tasks_step

echo "done."

if getopts "baseline" arg; then
  echo "Pre-paring data for baselines..."
  
  echo "Extracting UD trees that were actually used in STEP data (for exact comparison)..."
  python -m STEP.data_gen.extract_used_subcorpus
  
  echo "Simple STEP"
  python -m STEP.data_gen.create_ud_tasks_simple_step
  
  echo "T5+Dep Parse"
  python -m STEP.data_gen.create_ud_tasks_t5_plus_dep_parse
  echo "done."
fi
