local num_states = 60;

local name = "grammars/pt_data_step";


local train_data_path = name+"_train.pkl.xz";
local dev_data_path = name+"_dev.pkl.xz";
local easy_dev_data_path = name+"_easy_dev.pkl.xz";
local test_data_path = name+"_test.pkl.xz";


local tokenizer =   {
            f: "transformers.AutoTokenizer.from_pretrained",
                        pretrained_model_name_or_path: "t5-base"
                                };
                                
local data_loader(fname, batch_size) = {        
        "f": "load_ud_grammar_pickle",
        "batch_size": batch_size,
        "path": fname,
        "tokenizer": tokenizer,

} ;

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
        "f": "UDPretrainingModel",

        "num_nts": num_states,
        "num_functions": 30,
        
        "freeze_embeddings": true,

        "model": {
            f: "transformers.AutoModelForSeq2SeqLM.from_pretrained",
            pretrained_model_name_or_path: "t5-base"
            },
    },
    
    
    "tokenizer": tokenizer,

    "train_data_loader": data_loader(train_data_path, 16),
    "easy_validation_data_loader": data_loader(easy_dev_data_path, 32),
    "validation_data_loader": data_loader(dev_data_path, 32),

    "test_data_loader": data_loader(test_data_path, 32),
    
    "pretrain_data_loader": t5_data_loader(["00002"], 80, 50),
    
    "p_pretrain": 1.0,
    "use_aux_optimizer": true,

    "optimizer": {"[lazy]": "transformers.Adafactor", "scale_parameter": false, "relative_step": false,
                "warmup_init": false, "lr": 3e-4},
    "num_epochs": 1,
    
    "freq_save": 1,

    "logger": "[logger]",

    "num_accumulation_steps": 5,
    
    "save_checkpoints": false,

    "save_dir": "models/step_model",


   }

   ]
}

