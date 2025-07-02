# Code-switching Specific TTS Adaptations

This subdirectory contains copies of scripts from IMSToucan, adapted for code-switching. They are called using the [AnonFastSpeech2_codeswitching](../AnonFastSpeech2_codeswitching) script.

The changes include switching from language embedding per utterance to one language embedding for each phone, according to the language of that phone.
This is performed in [CSInferenceToucanTTS](CSInferenceToucanTTS.py). This gives us a matrix of language embeddings of size (sentence_length, emb_dim) instead of the previous vector (emb_dim).
The remaining scripts in this directory are adapted to support this matrix computation.

This subdirectory contains the following scripts:
* [CSInferenceToucanTTS](CSInferenceToucanTTS.py) based on [IMSToucan/IMSToucan/Modules/ToucanTTS/InferenceToucanTTS](../IMSToucan/Modules/ToucanTTS/InferenceToucanTTS.py)
* [CSConformer](CSConformer.py) based on [IMSToucan/Modules/GeneralLayers/Conformer](../IMSToucan/Modules/GeneralLayers/Conformer.py)
* [CSdit](CSdit.py) based on [IMSToucan/Modules/ToucanTTS/dit](../IMSToucan/Modules/ToucanTTS/dit.py)
* [CSdit_wrapper](CSdit_wrapper.py) based on [IMSToucan/Modules/ToucanTTS/dit_wrapper](../IMSToucan/Modules/ToucanTTS/dit_wrapper.py)
* [CSflow_matching](CSflow_matching.py) based on [IMSToucan/Modules/ToucanTTS/flow_matching](../IMSToucan/Modules/ToucanTTS/flow_matching.py)
* [CSutils](CSutils.py) based on parts in [IMSToucan/Modules/GeneralLayers/ConditionalLayerNorm](../IMSToucan/Modules/GeneralLayers/ConditionalLayerNorm.py)
