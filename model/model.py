from transformers import PreTrainedModel, WavLMModel
import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy_with_logits
from torchmetrics import PermutationInvariantTraining

class SED_Model(PreTrainedModel):
    def __init__(self, config, window = 1, stride = 1):
        super().__init__(config)

        self.criterion = PermutationInvariantTraining(binary_cross_entropy_with_logits, mode='speaker-wise', eval_func="min")

        # Window size and stride for average pooling
        self.window = window
        self.stride = stride

        # TODO: Input Normalization
        self.audio_extractor = WavLMModel.from_pretrained('microsoft/wavlm-base')

        self.avg_pool = nn.AvgPool1d(
            kernel_size = window,
            stride = stride
        )

        self.audio_classifier = nn.Linear(in_features=768, out_features=16)


    def forward(self, input_ids, labels=None):
            x = self.audio_extractor(input_ids)
            last_hidden_state = x.last_hidden_state
            x = self.avg_pool(last_hidden_state)
            logits = self.audio_classifier(x)

            batch, n_frames, _ = logits.shape
            logits = logits.reshape((batch, n_frames, 4, 4))
            
            if labels is not None:
                num_frames = logits.shape[1]
                labels = labels[:, :num_frames, :, :]

                # Change logits and labels shape suitable for criterion
                logits = logits.permute(0, 2, 3, 1)
                labels = labels.permute(0, 2, 3, 1)

                loss = self.criterion(logits, labels)

                return {"loss": loss}
            else:
                return logits
            
            
            
class SED_Model2(PreTrainedModel):
    def __init__(self, config, window=1, stride=1):
        super().__init__(config)


        self.window = window
        self.stride = stride
        self.criterion =PermutationInvariantTraining(binary_cross_entropy_with_logits, eval_func='min').to('cuda')

        self.audio_extractor = WavLMModel.from_pretrained('microsoft/wavlm-base')
        self.avg_pool = nn.AvgPool1d(
            kernel_size=window,
            stride=stride
        )
        self.audio_classifier = nn.Linear(in_features=768, out_features=16)# 1024

    def forward(self, input_values, labels=None):

        # Extract features from WavLM model
        x = self.audio_extractor(input_values)
        last_hidden_state = x.last_hidden_state


        x = self.avg_pool(last_hidden_state.transpose(1, 2))

        x = x.transpose(1, 2)


        logits = self.audio_classifier(x)
        size= logits.size()
        logits= logits.reshape(size[0],size[1],4,4)

        logits=logits.permute(0,2,3,1)

        loss = None
        if labels is not None:
            labels = labels[:,:,:, :logits.shape[3]]
            loss=self.criterion(logits, labels)
        return {"loss": loss, "logits": logits}            