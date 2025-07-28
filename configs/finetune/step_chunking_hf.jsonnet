local tokenizer =   {
            f: "transformers.AutoTokenizer.from_pretrained",
            pretrained_model_name_or_path: "t5-base"
        };

local task_train_loader = {
        "f": "prepare_task_dataset",

        "batch_size": 8,
        "path": std.extVar("train"),
        "tokenizer": tokenizer,
        lenient: true
};

local task_val_loader = {
        "f": "prepare_task_dataset",

        "batch_size": 16,
        "path":	std.extVar("test"),
        "tokenizer": tokenizer,
        lenient: true
};


{
  "random_seed": std.parseJson(std.extVar("seed")),
  "numpy_seed": std.parseJson(std.extVar("seed")),
  "pytorch_seed": std.parseJson(std.extVar("seed")),
  
  "imports": ["import transformers", "from STEP.sip_grammar import *",
  "from STEP.data_loading import *", "from STEP.task_finetune import *", "from STEP.chunk_eval import *"],

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
                pretrained_model_name_or_path: "namednil/STEP",
                trust_remote_code: true
             },

    "tokenizer": tokenizer,

    "train_data_loader": task_train_loader,
    "validation_data_loader": task_val_loader,
    "optimizer": {"[lazy]": "transformers.Adafactor", "scale_parameter": false, "relative_step": false,
                "warmup_init": false, "lr": 1e-4},
    "num_epochs": 80,
    
    "optimizer_groups": [
        [".*prefix_embedding.*", {"lr": std.parseJson(std.extVar("prefix_lr"))}],
        [".*", {"lr": std.parseJson(std.extVar("model_lr"))}]
    ],

    "logger": "[logger]",
    "eval_only_last_epochs": true,
    
    "num_accumulation_steps": 2,
    
    "custom_eval_on": {"f": "create_chunk_eval"},
    
    "batch_logging": true
  
   }
   
   ]
}
