local tokenizer =   {
            f: "transformers.AutoTokenizer.from_pretrained",
            pretrained_model_name_or_path: "t5-base"
        };

local task_train_loader = {
        "f": "prepare_task_dataset",

        "batch_size": 24,
        "path": std.extVar("train"),
        "tokenizer": tokenizer,
        lenient: true
};

local task_val_loader = {
        "f": "load_cogs_dataset",

        "batch_size": 48,
        "path":	std.extVar("valid"),
        "tokenizer": tokenizer
};

local task_test_loader = {
        "f": "load_cogs_dataset",

        "batch_size": 48,
        "path":	std.extVar("test"),
        "tokenizer": tokenizer
};


{
  "random_seed": std.parseJson(std.extVar("seed")),
  "numpy_seed": std.parseJson(std.extVar("seed")),
  "pytorch_seed": std.parseJson(std.extVar("seed")),
  
  "imports": ["import transformers", "from STEP.sip_grammar import *",
  "from STEP.data_loading import *", "from STEP.task_finetune import *", "from STEP.cogs_eval import *"],

  "logger": {
    f: "NeptuneLogger.create",
    "project": "<your-neptune-project>",
    "log_outputs": true,
  },
  "steps": [

   {
    "name": "finetune",
    "f": "finetune_model",

      model: {f: "transformers.AutoModelForSeq2SeqLM.from_pretrained",
                pretrained_model_name_or_path: std.extVar("load_model")
             },

    "tokenizer": tokenizer,

    "train_data_loader": task_train_loader,
    "validation_data_loader": task_val_loader,
    "test_data_loader": task_test_loader,
        "optimizer": {"[lazy]": "transformers.Adafactor", "scale_parameter": false, "relative_step": false,
                "warmup_init": false, "lr": std.parseJson(std.extVar("lr"))},
    "num_epochs": 50,
    
    "num_accumulation_steps": std.parseJson(std.extVar("acc")),
    
    "custom_eval_on": {"f": "create_cogs_eval_match"},

    "logger": "[logger]",
    
    "batch_logging": true
  
   }
   
   ]
}
