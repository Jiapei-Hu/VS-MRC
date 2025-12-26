import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat
from transformers import CLIPVisionModel, T5Config, AutoTokenizer, T5ForConditionalGeneration, logging, CLIPTokenizer, CLIPTextModel, T5Tokenizer#, InstructBlipVisionModel, InstructBlipQFormerModel
from copy import deepcopy
import random
from model.attention import Crossmodal_Attention


logging.set_verbosity_error()


class MVSAModel(nn.Module):
    def __init__(self, args) -> None:
        super(MVSAModel, self).__init__()
        self.args = args
        self.VitEncoder = CLIPVisionModel.from_pretrained(args.blip_name)
        self.VitEncoder.requires_grad_(False)

        self.clip_text_encoder = CLIPTextModel.from_pretrained(args.blip_name)
        self.clip_text_encoder.requires_grad_(False)

        VitDecoderLayer = nn.TransformerDecoderLayer(
            d_model=args.embed_dim,
            nhead=8,
            dim_feedforward=self.VitEncoder.config.intermediate_size,
            activation="gelu",
            batch_first=True,
        )
        self.VitDecoder = nn.TransformerDecoder(
            decoder_layer=VitDecoderLayer,
            num_layers=4,
        )

        self.query_tokens = nn.Parameter(
            torch.randn(1, args.query_nums, args.embed_dim))

        self.VitAdapter = nn.Linear(
            in_features=self.VitEncoder.config.hidden_size,
            out_features=args.embed_dim,
            bias=False,
        )

        self.T5 = T5ForConditionalGeneration.from_pretrained(args.t5_name)
        self.tokenizer = T5Tokenizer.from_pretrained(args.t5_name)

        self.dict_embeddings = nn.Parameter(torch.randn(1, args.dict_dim, args.embed_dim))

        self.dict_reconstruct = Crossmodal_Attention(
            num_attention_head=8,
            hidden_size=args.embed_dim,
            intermediate_size=args.embed_dim,
            dropout_prob=0.1,
            layer_norm_eps=1e-5
        )

        self.prompt_projection = nn.Linear(
            in_features=self.clip_text_encoder.config.hidden_size,
            out_features=self.VitEncoder.config.hidden_size
        )

        self.visual_linear_itm = nn.Linear(
            in_features=self.VitEncoder.config.hidden_size,
            out_features=768
        )
        self.textual_linear_itm = nn.Linear(
            in_features=self.clip_text_encoder.config.hidden_size,
            out_features=768
        )

        self.itm_head = nn.Linear(
            in_features=self.T5.config.d_model,
            out_features=2
        )



    def forward(
            self,
            input_ids,
            attention_mask,
            pixel_values,
            target_ids=None,
            target_attention_mask=None,
            labels=None,
            prompts=None
    ):
        bsz, img_len, channels, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(bsz*img_len, channels, height, width)

        vit_feats = self.VitEncoder(pixel_values=pixel_values) #[bs,257,1024]
        clip_visual_pooler = vit_feats[1]
        visual_feats = vit_feats[0][:, 1:]
        visual_feats = visual_feats.reshape(bsz, -1, self.VitEncoder.config.hidden_size)

        prompts = prompts.to(visual_feats.device)
        prompt_output = self.clip_text_encoder(input_ids=prompts['input_ids'], attention_mask=prompts['attention_mask'])
        clip_textual_pooler = prompt_output[1]
        prompt_embeds = self.prompt_projection(prompt_output[0])

        query_tokens = self.query_tokens.expand(bsz, -1, -1)

        visual_feats = self.VitAdapter(visual_feats)

        visual_query = self.VitDecoder(
            tgt=torch.cat((query_tokens, prompt_embeds), dim=1),
            memory=visual_feats,
        )[:,:self.args.query_nums]

        # reconstruct vision queries
        vision_dict_feature = self.dict_embeddings.expand(bsz, -1, -1)
        reconstruct_query = self.dict_reconstruct(
            query=visual_query,
            key=vision_dict_feature,
            value=vision_dict_feature
        )

        textual_embeds = self.T5.get_input_embeddings()(input_ids)

        clip_visual_pooler = clip_visual_pooler.reshape(bsz, 4, -1)
        clip_visual_pooler = torch.mean(clip_visual_pooler, dim=1)
        visual_cls = F.normalize(self.visual_linear_itm(clip_visual_pooler), dim=-1)
        textual_cls = F.normalize(self.textual_linear_itm(clip_textual_pooler), dim=-1)
        itm_sim = torch.matmul(visual_cls, textual_cls.t())

        with torch.no_grad():
            itm_sim = F.softmax(itm_sim, dim=1)
            itm_sim.fill_diagonal_(0)

        # select a negative text for each image
        text_embeds_neg = []
        # text_atts_neg = []
        for b in range(bsz):
            neg_idx = torch.multinomial(itm_sim[b], 1).item()
            text_embeds_neg.append(textual_embeds[neg_idx])
            # text_atts_neg.append(text.attention_mask[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        # text_atts_neg = torch.stack(text_atts_neg,dim=0)

        text_embeds_all = torch.cat((textual_embeds, text_embeds_neg), dim=0)
        image_embeds_all = torch.cat((reconstruct_query, reconstruct_query), dim=0)
        itm_embeds_all = torch.cat((image_embeds_all, text_embeds_all), dim=1)
        itm_labels = torch.cat([torch.ones(bsz, dtype=torch.long), torch.zeros(bsz, dtype=torch.long)],
                               dim=0).to(visual_feats.device)
        # 创建一个打乱的索引
        indices = list(range(len(itm_embeds_all)))
        random.shuffle(indices)

        # 使用索引来重新排列两个张量
        shuffled_embeds = itm_embeds_all[indices]
        shuffled_labels = itm_labels[indices]

        with torch.no_grad():
            itm_hidden_state = self.T5.encoder(
                inputs_embeds=shuffled_embeds
            )[0].float()

        itm_outputs = self.itm_head(F.normalize(torch.mean(itm_hidden_state, dim=1), dim=-1))
        loss_itm = F.cross_entropy(itm_outputs, shuffled_labels)
        self.loss_itm = loss_itm*0.1

        combine_embeds = torch.cat((reconstruct_query, textual_embeds), dim=1)

        outputs = self.T5(
            inputs_embeds=combine_embeds,
            labels=target_ids
        )

        #---------------------------------------------End----------------------------------------------------#
        return outputs['loss'], outputs['logits'], 0, 0, self.loss_itm

    def generate(
            self,
            input_ids,
            attention_mask,
            pixel_values,
            target_ids,
            batch_labels,
            prompts
    ):

        bsz, img_len, channels, height, width = pixel_values.shape
        pixel_values = pixel_values.reshape(bsz*img_len, channels, height, width)

        vit_feats = self.VitEncoder(pixel_values=pixel_values)[0] #[bs,257,1024]
        visual_feats = vit_feats[:, 1:]
        visual_feats = visual_feats.reshape(bsz, -1, self.VitEncoder.config.hidden_size)

        prompts = prompts.to(vit_feats.device)
        prompt_output = self.clip_text_encoder(input_ids=prompts['input_ids'], attention_mask=prompts['attention_mask'])
        prompt_embeds = self.prompt_projection(prompt_output[0])

        query_tokens = self.query_tokens.expand(bsz, -1, -1)

        visual_feats = self.VitAdapter(visual_feats)

        visual_query = self.VitDecoder(
            tgt=torch.cat((query_tokens, prompt_embeds), dim=1),
            memory=visual_feats,
        )[:,:self.args.query_nums]


        vision_dict_feature = self.dict_embeddings.expand(bsz, -1, -1)
        reconstruct_query = self.dict_reconstruct(
            query=visual_query,
            key=vision_dict_feature,
            value=vision_dict_feature
        )

        textual_embeds = self.T5.get_input_embeddings()(input_ids)
        combine_embeds = torch.cat((reconstruct_query, textual_embeds), dim=1)

        outs = self.T5.generate(
            inputs_embeds=combine_embeds,
            max_length=100,
            do_sample=self.args.do_sample,
            temperature=0.2 if self.args.do_sample else 1.0,
            top_p=0.7 if self.args.do_sample else 1.0,
            top_k=30 if self.args.do_sample else 50,
            num_return_sequences=1,
            num_beams=1
        )

        dec = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in outs]
        target = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in target_ids]

        # return dec, batch_labels
        return dec, target

    def save_weights(self, file_path):
        torch.save(self.state_dict(), file_path)
        print(f"Model weights saved to {file_path}")

    def load_weights(self, file_path, device):
        self.load_state_dict(torch.load(file_path, map_location=device))
        self.to(device)
        print(f"Model weights loaded from {file_path}")


