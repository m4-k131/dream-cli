"""TF2/Keras model registry for dreamer2.

Provides a generic feature-extractor builder that wraps any
``tf.keras.applications`` model and returns a model whose outputs are
the activations of a user-chosen set of layers.
"""
from __future__ import annotations

from typing import Any, Callable

import tensorflow as tf

# ---------------------------------------------------------------------------
# Registry: name → (constructor_fn, preprocess_fn)
# Add new entries here to expose additional architectures.
# ---------------------------------------------------------------------------
_REGISTRY: dict[str, tuple[Callable[..., tf.keras.Model], Callable]] = {
    "InceptionV3": (
        tf.keras.applications.InceptionV3,
        tf.keras.applications.inception_v3.preprocess_input,
    ),
    "InceptionResNetV2": (
        tf.keras.applications.InceptionResNetV2,
        tf.keras.applications.inception_resnet_v2.preprocess_input,
    ),
    "VGG16": (
        tf.keras.applications.VGG16,
        tf.keras.applications.vgg16.preprocess_input,
    ),
    "VGG19": (
        tf.keras.applications.VGG19,
        tf.keras.applications.vgg19.preprocess_input,
    ),
    "ResNet50": (
        tf.keras.applications.ResNet50,
        tf.keras.applications.resnet.preprocess_input,
    ),
    "ResNet50V2": (
        tf.keras.applications.ResNet50V2,
        tf.keras.applications.resnet_v2.preprocess_input,
    ),
    "EfficientNetB0": (
        tf.keras.applications.EfficientNetB0,
        tf.keras.applications.efficientnet.preprocess_input,
    ),
    "EfficientNetB4": (
        tf.keras.applications.EfficientNetB4,
        tf.keras.applications.efficientnet.preprocess_input,
    ),
    "MobileNetV2": (
        tf.keras.applications.MobileNetV2,
        tf.keras.applications.mobilenet_v2.preprocess_input,
    ),
    "DenseNet121": (
        tf.keras.applications.DenseNet121,
        tf.keras.applications.densenet.preprocess_input,
    ),
    "NASNetLarge": (
        tf.keras.applications.NASNetLarge,
        tf.keras.applications.nasnet.preprocess_input,
    ),
    "Xception": (
        tf.keras.applications.Xception,
        tf.keras.applications.xception.preprocess_input,
    ),
}

# Default layer names per model (good starting choices for DeepDream).
_DEFAULT_LAYERS: dict[str, list[str]] = {
    "InceptionV3": ["mixed3", "mixed5"],
    "InceptionResNetV2": ["mixed_5b", "mixed_6a"],
    "VGG16": ["block3_conv1", "block4_conv1"],
    "VGG19": ["block3_conv1", "block4_conv1"],
    "ResNet50": ["conv3_block4_out", "conv4_block6_out"],
    "ResNet50V2": ["conv3_block4_1_relu", "conv4_block6_1_relu"],
    "EfficientNetB0": ["block3a_expand_activation", "block5a_expand_activation"],
    "EfficientNetB4": ["block3a_expand_activation", "block5a_expand_activation"],
    "MobileNetV2": ["block_6_expand_relu", "block_13_expand_relu"],
    "DenseNet121": ["pool3_relu", "pool4_relu"],
    "NASNetLarge": ["reduction_concat_stem_1", "normal_concat_2"],
    "Xception": ["block3_sepconv1_act", "block8_sepconv1_act"],
}


def known_model_names() -> list[str]:
    """Return the list of registered model names."""
    return list(_REGISTRY.keys())


def default_layers_for(model_name: str) -> list[str]:
    """Return the default layer list for *model_name*."""
    if model_name not in _DEFAULT_LAYERS:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {known_model_names()}"
        )
    return list(_DEFAULT_LAYERS[model_name])


def list_layer_names(model_name: str) -> list[str]:
    """Return all layer names in the base model (no weights loaded into memory fully).

    Useful for interactive layer selection.
    """
    if model_name not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {known_model_names()}"
        )
    constructor, _ = _REGISTRY[model_name]
    base = constructor(include_top=False, weights=None)
    return [layer.name for layer in base.layers]


def load_feature_extractor(
    model_name: str,
    layer_names: list[str] | None = None,
    weights: str = "imagenet",
    **constructor_kwargs: Any,
) -> tuple[tf.keras.Model, Callable]:
    """Build a feature-extraction model and return ``(feature_model, preprocess_fn)``.

    Parameters
    ----------
    model_name:
        One of the keys in ``known_model_names()``.
    layer_names:
        Names of layers whose outputs form the model outputs.  When *None*,
        the default set for *model_name* is used.
    weights:
        Passed verbatim to the Keras constructor (default ``"imagenet"``).
    **constructor_kwargs:
        Extra keyword arguments forwarded to the Keras constructor
        (e.g. ``input_shape``, ``include_top``).

    Returns
    -------
    feature_model
        A ``tf.keras.Model`` that accepts a pre-processed image tensor and
        returns a list of activation tensors, one per requested layer.
    preprocess_fn
        The matching preprocessing function (accepts ``np.ndarray`` in
        ``[0, 255]`` uint8 range, returns float tensor).
    """
    if model_name not in _REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {known_model_names()}"
        )
    constructor, preprocess_fn = _REGISTRY[model_name]

    if layer_names is None:
        layer_names = _DEFAULT_LAYERS[model_name]

    base_model: tf.keras.Model = constructor(
        include_top=constructor_kwargs.pop("include_top", False),
        weights=weights,
        **constructor_kwargs,
    )

    outputs = []
    missing = []
    for name in layer_names:
        try:
            outputs.append(base_model.get_layer(name).output)
        except ValueError:
            missing.append(name)
    if missing:
        available = [layer.name for layer in base_model.layers]
        raise ValueError(
            f"Layer(s) not found in {model_name}: {missing}.\n"
            f"Available layers: {available}"
        )

    feature_model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
    feature_model.trainable = False
    return feature_model, preprocess_fn
