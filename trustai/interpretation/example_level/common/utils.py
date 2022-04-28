"""
Some useful functions.
"""

def get_sublayer(model, sublayer_name='classifier'):
    """
    Get the sublayer named sublayer_name in model.
    Args:
        model (obj:`paddle.nn.Layer`): Any paddle model.
        sublayer_name (obj:`str`, defaults to classifier): The sublayer name.
    Returns:
        layer(obj:`paddle.nn.Layer.common.sublayer_name`):The sublayer named sublayer_name in model. 
    """
    for name, layer in model.named_children():
        if name == sublayer_name:
            return layer