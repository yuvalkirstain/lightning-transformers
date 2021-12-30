#!/bin/bash -x

for n_distractors in 0 16 64
do
  for n_train_samples in 1024 4096 16384 65536
  do
    python send_experiments.py \
     backbone.pretrained_model_name_or_path=allenai/longformer-base-4096 \
     task=nlp/question_answering dataset=nlp/question_answering/squad \
     task.cfg.monitor=f1 \
     task.cfg.mode=max \
     dataset.cfg.dataset_name=null \
     dataset.cfg.train_file="/home/olab/kirstain/lightning-transformers/data/squad_with_distracting_contexts/${n_distractors}/train.json" \
     dataset.cfg.validation_file="/home/olab/kirstain/lightning-transformers/data/squad_with_distracting_contexts/${n_distractors}/validation.json" \
     dataset.cfg.load_from_cache_file=false dataset.cfg.max_length=4096 \
     dataset.cfg.limit_train_samples=${n_train_samples} \
     scheduler.num_warmup_steps=500  \
     trainer.gpus=1 \
     trainer.check_val_every_n_epoch=1  \
     trainer.num_sanity_val_steps=0 \
     trainer.accumulate_grad_batches=16 \
     trainer.logger.project="squad_with_distracting_contexts" \
     trainer.logger.name="${n_distractors}-${n_train_samples}" \
     trainer.flush_logs_every_n_steps=50 \
     trainer.log_every_n_steps=100 \
     training.batch_size=2 \
     slurm.run_name="squad-${n_distractors}-${n_train_samples}" \
     slurm.time=1440 \
     slurm.constraint=geforce_rtx_3090
     sleep 1
  done
done