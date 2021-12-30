#!/bin/bash -x

for n_train_samples in 1024 4096 16384 65536
do
  python send_experiments.py \
 backbone.pretrained_model_name_or_path=allenai/longformer-base-4096 \
 task=nlp/question_answering dataset=nlp/question_answering/squad \
 task.cfg.monitor=f1 \
 task.cfg.mode=max \
 dataset.cfg.dataset_name="yuvalkirstain/squad_full_doc" \
 dataset.cfg.load_from_cache_file=false dataset.cfg.max_length=4096 \
 dataset.cfg.limit_train_samples=${n_train_samples} \
 scheduler.num_warmup_steps=500  \
 trainer.gpus=1 \
 trainer.check_val_every_n_epoch=1  \
 trainer.num_sanity_val_steps=0 \
 trainer.accumulate_grad_batches=4 \
 trainer.logger.project="squad_with_full_doc" \
 trainer.logger.name="full_doc-${n_train_samples}" \
 trainer.flush_logs_every_n_steps=50 \
 trainer.log_every_n_steps=100 \
 training.batch_size=8 \
 slurm.run_name="squad-full-doc-${n_train_samples}" \
 slurm.time=1440 \
 slurm.nodelist=n-401
 sleep 1
done