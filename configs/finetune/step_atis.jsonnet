local tokenizer =   {
            f: "transformers.AutoTokenizer.from_pretrained",
            pretrained_model_name_or_path: "t5-base"
        };

local task_train_loader = {
        "f": "prepare_task_dataset",

        "batch_size": 16,
        "path": std.extVar("train"),
        "tokenizer": tokenizer,
        lenient: true
};

local task_val_loader = {
        "f": "prepare_task_dataset",

        "batch_size": 32,
        "path":	std.extVar("valid"),
        "tokenizer": tokenizer,
        lenient: true
};

local task_test_loader = {
        "f": "prepare_task_dataset",

        "batch_size": 32,
        "path":	std.extVar("test"),
        "tokenizer": tokenizer,
        lenient: true
};


{
  "random_seed": std.parseJson(std.extVar("seed")),
  "numpy_seed": std.parseJson(std.extVar("seed")),
  "pytorch_seed": std.parseJson(std.extVar("seed")),
  
  
  "imports": ["import transformers", "from STEP.sip_grammar import *",
  "from STEP.data_loading import *", "from STEP.task_finetune import *", "from STEP.atis_eval import *"],

  "logger": {
    f: "NeptuneLogger.create",
    "project": "<your-neptune-project>",
    "log_outputs": true,
  },
  "steps": [

   {
    "name": "finetune",
    "f": "finetune_model",
    
    "model": {
        f: "StructuredPrefixEmbeddingModelForCFG.from_sip_pretrained",
        "prefix_length": 10,
        "path": std.extVar("load_model"),
        
        "data_loader": {
            "f": "load_ud_grammar_pickle",
            "batch_size": 512,
            "path": std.extVar("load_prefix"),
            "tokenizer": tokenizer
        }
        
    },

    "tokenizer": tokenizer,

    "train_data_loader": task_train_loader,
    "validation_data_loader": task_val_loader,
    "test_data_loader": task_test_loader,
    "optimizer": {"[lazy]": "transformers.Adafactor", "scale_parameter": false, "relative_step": false,
                "warmup_init": false, "lr": 1e-4},
    "num_epochs": std.parseJson(std.extVar("epoch")),
    
    "num_accumulation_steps": std.parseJson(std.extVar("acc")),
    
    "optimizer_groups": [
        [".*prefix_embedding.*", {"lr": std.parseJson(std.extVar("prefix_lr"))}],
        [".*", {"lr": std.parseJson(std.extVar("model_lr"))}]
    ],

    "eval_metric": "+tree_acc",

    "logger": "[logger]",
    
    "custom_eval_on": {"f": "create_atis_eval"},
    
    "batch_logging": true
  
   }
   
   ]
}