from diffusers.models.attention_processor import AttnProcessor2_0

from diffusers.models.attention import Attention
import math

class JointAttnProcessor:
    """
    Attention processor for SD3-like self-attention projections.
    This processor handles both self-attention and cross-attention mechanisms.
    """

    def __init__(self):
        pass

    def __call__(
        self,
        attn,  # Attention module
        hidden_states: torch.FloatTensor,  # Input tensor [batch_size, seq_len, hidden_dim]
        encoder_hidden_states: torch.FloatTensor = None,  # Optional context tensor [batch_size, context_len, hidden_dim]
        attention_mask = None,  # Optional mask tensor [batch_size, seq_len]
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        # Store residual for skip connection
        residual = hidden_states  # [batch_size, seq_len, hidden_dim]
        batch_size = hidden_states.shape[0]

        # Project input into query, key, value spaces
        # Each projection: [batch_size, seq_len, hidden_dim]
        query = attn.to_q(hidden_states)
        key = attn.to_k(hidden_states)
        value = attn.to_v(hidden_states)

        # Reshape for multi-head attention
        inner_dim = key.shape[-1]  # Total dimension across all heads
        head_dim = inner_dim // attn.heads  # Dimension per attention head

        # Reshape tensors to [batch_size, n_heads, seq_len, head_dim]
        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # Apply normalization if specified
        if attn.norm_q is not None:
            query = attn.norm_q(query)
        if attn.norm_k is not None:
            key = attn.norm_k(key)
        
        # Handle cross-attention if encoder_hidden_states is provided
        if encoder_hidden_states is not None:
            # Project encoder states to query, key, value
            # [batch_size, context_len, hidden_dim] -> [batch_size, n_heads, context_len, head_dim]
            encoder_hidden_states_query_proj = attn.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = attn.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = attn.add_v_proj(encoder_hidden_states)

            # Reshape encoder projections
            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, attn.heads, head_dim
            ).transpose(1, 2)

            # Apply normalization to encoder projections if specified
            if attn.norm_added_q is not None:
                encoder_hidden_states_query_proj = attn.norm_added_q(encoder_hidden_states_query_proj)
            if attn.norm_added_k is not None:
                encoder_hidden_states_key_proj = attn.norm_added_k(encoder_hidden_states_key_proj)

            # Concatenate self and cross attention tensors
            # [batch_size, n_heads, seq_len + context_len, head_dim]
            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        # Compute scaled dot-product attention
        L, S = query.size(-2), key.size(-2)  # L: target sequence length, S: source sequence length
        scale_factor = 1 / math.sqrt(query.size(-1))  # Scaling factor for numerical stability
        attn_bias = torch.zeros(L, S, dtype=query.dtype, device=query.device)

        # Apply attention mask if provided
        if attention_mask is not None:
            if attention_mask.dtype == torch.bool:
                attn_bias.masked_fill_(attention_mask.logical_not(), float("-inf"))
            else:
                attn_bias += attention_mask

        # Compute attention weights and apply them to values
        # [batch_size, n_heads, seq_len, seq_len]
        attn_weight = query @ key.transpose(-2, -1) * scale_factor
        attn_weight += attn_bias
        attn_weight = torch.softmax(attn_weight, dim=-1)
        attn_weight = torch.dropout(attn_weight, 0.0, train=False)
        print("attn_weight: ", attn_weight.shape)
        # [batch_size, n_heads, seq_len, head_dim]
        hidden_states = attn_weight @ value
        print("hidden_states", hidden_states.shape)
        # Reshape back to original dimensions
        # [batch_size, seq_len, hidden_dim]
        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # Handle cross-attention output
        if encoder_hidden_states is not None:
            # Split self-attention and cross-attention outputs
            hidden_states, encoder_hidden_states = (
                hidden_states[:, : residual.shape[1]],
                hidden_states[:, residual.shape[1] :],
            )
            if not attn.context_pre_only:
                encoder_hidden_states = attn.to_add_out(encoder_hidden_states)

        # Final linear projection and dropout
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        print("hidden_states final", hidden_states.shape)
        print("====")
        # Return appropriate output based on whether cross-attention was used
        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states
        else:
            return hidden_states
joint_attn_processor = JointAttnProcessor()
def mod_forward_sd3(
    self,
    hidden_states: torch.Tensor,
    encoder_hidden_states = None,
    attention_mask = None,
    **cross_attention_kwargs,
) -> torch.Tensor:
    r"""
    The forward method of the `Attention` class.

    Args:
        hidden_states (`torch.Tensor`):
            The hidden states of the query.
        encoder_hidden_states (`torch.Tensor`, *optional*):
            The hidden states of the encoder.
        attention_mask (`torch.Tensor`, *optional*):
            The attention mask to use. If `None`, no mask is applied.
        **cross_attention_kwargs:
            Additional keyword arguments to pass along to the cross attention.

    Returns:
        `torch.Tensor`: The output of the attention layer.
    """
    # The `Attention` class can call different attention processors / attention functions
    # here we simply pass along all tensors to the selected processor class
    # For standard processors that are defined here, `**cross_attention_kwargs` is empty

    attn_parameters = set(inspect.signature(self.processor.__call__).parameters.keys())
    quiet_attn_parameters = {"ip_adapter_masks"}
    unused_kwargs = [
        k for k, _ in cross_attention_kwargs.items() if k not in attn_parameters and k not in quiet_attn_parameters
    ]

    cross_attention_kwargs = {k: w for k, w in cross_attention_kwargs.items() if k in attn_parameters}
    if isinstance(self.processor, AttnProcessor2_0):
        pass
    else:
        self.processor = joint_attn_processor
    return self.processor(
        self,
        hidden_states,
        encoder_hidden_states=encoder_hidden_states,
        attention_mask=attention_mask,
        **cross_attention_kwargs,
    )
for _module in pipe.transformer.modules():
    if _module.__class__.__name__ == "Attention":
        _module.__class__.__call__ = mod_forward_sd3