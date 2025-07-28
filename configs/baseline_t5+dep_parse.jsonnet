local name = "grammars/pt_data_t5_plus_dep_parse";


local train_data_path = name+"_train.tsv";
local test_data_path = name+"_test.tsv";


local tokenizer =   {
            f: "transformers.AutoTokenizer.from_pretrained",
                        pretrained_model_name_or_path: "t5-base"
                                };

local data_loader(fname, batch_size) = {
        "f": "prepare_task_dataset",

        "batch_size": batch_size,
        "path":	fname,
        "tokenizer": tokenizer,
        lenient: true
};


local t5_data_loader(chunk_id, max_len, batch_size) = {
  "f": "load_c4_chunk",
  "batch_size": batch_size,
  "chunk_id": chunk_id,
  "input_length": max_len,
  "tokenizer": tokenizer,
};


{
  "imports": ["import transformers", "from STEP.sip_grammar import *",
   "from STEP.data_loading import *", "from STEP.pretraining import *", "from STEP.t5_denoising import *"],

  "logger": {
    f: "NeptuneLogger.create",
    "project": "<your-neptune-project>"
  },

  "steps": [

   {
    "name": "pretrain",
    "f": "pretrain",


    "model": {
        f: "transformers.AutoModelForSeq2SeqLM.from_pretrained",
        pretrained_model_name_or_path: "t5-base"
        },
    
    "tokenizer": tokenizer,

    "train_data_loader": data_loader(train_data_path, 6),
    "easy_validation_data_loader": null,
    "validation_data_loader": null,

    "test_data_loader": data_loader(test_data_path, 12),
    
    "pretrain_data_loader": t5_data_loader(["00002", "00003", "00004", "00005", "00006"], 80, 50),
    
    "p_pretrain": 1.0,
    "use_aux_optimizer": true,

    "optimizer": {"[lazy]": "transformers.Adafactor", "scale_parameter": false, "relative_step": false,
                "warmup_init": false, "lr": 3e-4},
    "num_epochs": 2,
    
    "freq_save": 1,

    "logger": "[logger]",

    "num_accumulation_steps": 8,
    
    "save_checkpoints": true,

    "save_dir": "models/baseline_t5_plus_dep_parse",


   }

   ]
}

