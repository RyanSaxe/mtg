import tensorflow as tf
import sys
from tqdm import tqdm
import numpy as np

class Trainer:
    def __init__(
        self,
        model,
        generator=None,
        val_generator=None,
        features=None,
        target=None,
        weights = None,
        val_features = None,
        val_target = None,
        val_weights = None,
        clip = 5.0,
    ):
        self.generator=generator
        self.val_generator=val_generator
        self.features = features
        self.target = target
        self.model = model
        self.epoch_n = 0
        self.clip = clip
        if self.target is not None:
            self.batch_ids = np.arange(len(self.target))
        else:
            self.batch_ids = None
        self.weights = weights
        self.val_features = val_features
        self.val_target = val_target
        self.val_weights = val_weights

        if self.generator is not None:
            assert self.features is None
            assert self.target is None
            assert self.weights is None
        else:
            assert self.features is not None
            assert self.target is not None

        if self.val_generator is not None:
            assert self.generator is not None
            assert self.val_features is None
            assert self.val_target is None
            assert self.val_weights is None
    
    def _step(self, batch_features, batch_target, batch_weights):
        with tf.GradientTape() as tape:
            output = self.model(batch_features, training=True)
            loss = self.model.loss(batch_target, output, sample_weight=batch_weights)
            metrics = self.model.compute_metrics(batch_target, output, sample_weight=batch_weights)
            #put regularization here if necessary
        grads = tape.gradient(loss, self.model.trainable_variables)
        if self.clip:
            grads, _ = tf.clip_by_global_norm(grads, self.clip)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, metrics
    def train(self, n_epochs, batch_size=32, verbose=True, print_keys=[]):
        n_batches = len(self.batch_ids) // batch_size if self.generator is None else len(self.generator)
        end_at = self.epoch_n + n_epochs
        for _ in range(n_epochs):
            self.epoch_n += 1
            if self.batch_ids is not None:
                np.random.shuffle(self.batch_ids)
            if verbose:
                progress = tqdm(
                    total = n_batches,
                    desc = f'Epoch {self.epoch_n}/{end_at}',
                    unit = 'Batch'
                )
            extras = {k:[] for k in print_keys}
            losses = []
            val_losses = []
            has_val = self.val_generator is not None or self.val_features is not None
            extra_metric_keys = ['top1','top2','top3']
            if has_val:
                extra_metric_keys += ['val_top1', 'val_top2', 'val_top3']
            extra_metrics = {k:[] for k in extra_metric_keys}
            for i in range(n_batches):
                val_loss = None
                if self.generator is None:
                    batch_idx = self.batch_ids[i * batch_size:(i+1) * batch_size]
                    batch_features = self.features[batch_idx,:]
                    batch_target = self.target[batch_idx,:]
                    if self.weights is not None:
                        batch_weights = self.weights[batch_idx]
                        batch_weights = batch_weights/batch_weights.sum()
                    else:
                        batch_weights = None
                else:
                    batch_features, batch_target, batch_weights = self.generator[i]
                loss, metrics = self._step(batch_features, batch_target, batch_weights)
                extra_metrics['top1'].append(metrics[0])
                extra_metrics['top2'].append(metrics[1])
                extra_metrics['top3'].append(metrics[2])
                losses.append(np.average(loss))
                for attr_name in extras.keys():
                    attr = getattr(self.model, attr_name, None)
                    extras[attr_name].append(attr)
                
                if self.val_generator is not None:
                    val_features, val_target, val_weights = self.val_generator[i]
                    # must get attention here to serialize the input for saving
                    val_output = self.model(val_features, training=False)
                    val_loss = self.model.loss(val_target, val_output, sample_weight=val_weights)
                    val_metrics = self.model.compute_metrics(val_target, val_output, sample_weight=val_weights)
                    extra_metrics['val_top1'].append(val_metrics[0])
                    extra_metrics['val_top2'].append(val_metrics[1])
                    extra_metrics['val_top3'].append(val_metrics[2])
                    val_losses.append(np.average(val_loss))
                if verbose:
                    extra_to_show = {
                        **{k:np.average(v) for k,v in extras.items()},
                        **{k:np.average(v) for k,v in extra_metrics.items()}
                    }                        
                    if len(val_losses) > 0:
                        progress.set_postfix(loss=np.average(losses), val_loss=np.average(val_losses), **extra_to_show)
                    else:
                        progress.set_postfix(loss=np.average(losses), **extra_to_show)
                    progress.update(1)
            if verbose:
                #run model as if not training on validation data to get out of sample performance
                if self.val_features is not None:
                    val_out = self.model(self.val_features, training=False)
                    val_loss = self.model.loss(self.val_target, val_out, sample_weight=self.val_weights)
                    progress.set_postfix(loss=np.average(losses), val_loss=np.average(val_loss), **extra_to_show)
                progress.close()
            if self.generator is not None:
                self.generator.on_epoch_end()
            if self.val_generator is not None:
                self.val_generator.on_epoch_end()