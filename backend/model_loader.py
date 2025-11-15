"""
Model loader module for the bias detection model.
Handles loading and inference of the fine-tuned ModernBERT/DeBERTa and Longformer models.
Automatically switches to Longformer for articles exceeding 512 tokens.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel
import numpy as np
from typing import Tuple, Dict
import os


class BiasClassifier:
    """
    Wrapper class for the fine-tuned bias classification models.
    Automatically uses DeBERTa for short articles and Longformer for long articles.
    """
    
    def __init__(self, 
                 short_model_path: str = "./models/deberta_model",
                 long_model_path: str = "./models/longformer-finetuned-model",
                 token_threshold: int = 512):
        """
        Initialize the bias classifier with both short and long models.
        
        Args:
            short_model_path: Path to the short-article model (DeBERTa/ModernBERT)
            long_model_path: Path to the long-article model (Longformer)
            token_threshold: Token count threshold to switch models (default: 512)
        """
        self.short_model_path = short_model_path
        self.long_model_path = long_model_path
        self.token_threshold = token_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 3-label system for short model
        self.label_map_3 = {0: "Left", 1: "Centre", 2: "Right"}
        
        # 5-label system for long model (Longformer)
        self.label_map_5 = {
            0: "left",
            1: "lean left", 
            2: "center",
            3: "lean right",
            4: "right"
        }
        
        # Load both models
        self._load_models()
    
    def _load_models(self):
        """Load both the short-article and long-article models."""
        try:
            # Load short model (DeBERTa/ModernBERT with LoRA)
            print(f"Loading short-article model from {self.short_model_path}...")
            
            if not os.path.exists(self.short_model_path):
                raise FileNotFoundError(f"Short model directory not found: {self.short_model_path}")
            
            # Load short model tokenizer
            self.short_tokenizer = AutoTokenizer.from_pretrained(self.short_model_path)
            
            # Check if it's a PEFT model
            adapter_config_path = os.path.join(self.short_model_path, "adapter_config.json")
            
            if os.path.exists(adapter_config_path):
                print("Short model is a PEFT model, loading with LoRA adapters...")
                import json
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                    base_model_name = adapter_config.get('base_model_name_or_path', 'answerdotai/ModernBERT-base')
                
                # Load base model
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    base_model_name,
                    num_labels=3
                )
                
                # Load LoRA adapter
                self.short_model = PeftModel.from_pretrained(base_model, self.short_model_path)
            else:
                print("Loading short model as standard model...")
                self.short_model = AutoModelForSequenceClassification.from_pretrained(
                    self.short_model_path,
                    num_labels=3
                )
            
            self.short_model = self.short_model.to(self.device)
            self.short_model.eval()
            print(f"Short model loaded successfully on {self.device}")
            
            # Load long model (Longformer)
            print(f"\nLoading long-article model from {self.long_model_path}...")
            
            if not os.path.exists(self.long_model_path):
                raise FileNotFoundError(f"Long model directory not found: {self.long_model_path}")
            
            # Load long model tokenizer and model
            self.long_tokenizer = AutoTokenizer.from_pretrained(self.long_model_path)
            self.long_model = AutoModelForSequenceClassification.from_pretrained(
                self.long_model_path,
                num_labels=5
            )
            
            self.long_model = self.long_model.to(self.device)
            self.long_model.eval()
            print(f"Long model loaded successfully on {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load models: {str(e)}")
    
    def _map_5_to_3_labels(self, probs_5: np.ndarray) -> Dict[str, float]:
        """
        Map 5-label probabilities to 3-label system.
        
        5-label: left (0), lean left (1), center (2), lean right (3), right (4)
        3-label: Left, Centre, Right
        
        Mapping strategy:
        - Left = left + lean left
        - Centre = center
        - Right = lean right + right
        
        Args:
            probs_5: Probability array of shape (5,) from Longformer
            
        Returns:
            Dictionary with Left, Centre, Right probabilities
        """
        # Sum probabilities
        left_prob = float(probs_5[0] + probs_5[1])  # left + lean left
        centre_prob = float(probs_5[2])              # center
        right_prob = float(probs_5[3] + probs_5[4]) # lean right + right
        
        # Normalize to ensure they sum to 1.0
        total = left_prob + centre_prob + right_prob
        if total > 0:
            left_prob /= total
            centre_prob /= total
            right_prob /= total
        
        return {
            "Left": left_prob,
            "Centre": centre_prob,
            "Right": right_prob
        }
    
    def predict(self, text: str, max_length: int = 512, get_attributions: bool = True) -> Dict[str, any]:
        """
        Predict the bias classification for the given text.
        Automatically uses short model (DeBERTa) for ≤512 tokens, Longformer for longer texts.
        
        Args:
            text: Input article text
            max_length: Maximum token length for the short model (512 for DeBERTa)
            get_attributions: Whether to compute SHAP word importance (slower, only for short model)
            
        Returns:
            Dictionary containing:
                - label: Predicted bias label (Left, Centre, Right)
                - confidence: Confidence score (0-1)
                - probabilities: Dictionary of probabilities for each class
                - model_used: Which model was used ("short" or "long")
                - num_tokens: Number of tokens in the input
                - word_importance: SHAP attributions (if get_attributions=True and short model used)
        """
        try:
            # Check text length with short tokenizer
            test_tokens = self.short_tokenizer(
                text,
                return_tensors="pt",
                truncation=False,
                add_special_tokens=True
            )
            num_tokens = test_tokens['input_ids'].shape[1]
            
            # Decide which model to use
            if num_tokens <= self.token_threshold:
                # Use short model (DeBERTa/ModernBERT)
                print(f"Using short model (DeBERTa) - {num_tokens} tokens")
                result = self._predict_short(text, max_length, get_attributions=get_attributions)
                result["model_used"] = "short"
                result["num_tokens"] = num_tokens
            else:
                # Use long model (Longformer)
                print(f"Using long model (Longformer) - {num_tokens} tokens (exceeds {self.token_threshold})")
                result = self._predict_long(text, get_attributions=False)  # No attributions for long model
                result["model_used"] = "long"
                result["num_tokens"] = num_tokens
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Prediction failed: {str(e)}")
    
    def _predict_short(self, text: str, max_length: int, get_attributions: bool = True) -> Dict[str, any]:
        """
        Predict using the short model (DeBERTa/ModernBERT).
        
        Args:
            text: Input text
            max_length: Maximum token length
            get_attributions: Whether to compute word importance scores
            
        Returns:
            Prediction dictionary
        """
        # Tokenize input
        inputs = self.short_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.short_model(**inputs)
            logits = outputs.logits
            
            # Apply softmax to get probabilities
            probabilities = torch.nn.functional.softmax(logits, dim=-1)
            probabilities = probabilities.cpu().numpy()[0]
            
            # Get predicted class and confidence
            predicted_class = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_class])
        
        # Get word importance scores for each class
        word_importance = None
        if get_attributions:
            word_importance = self._get_word_attributions_short(text, inputs, max_length)
        
        # Prepare result
        result = {
            "label": self.label_map_3[predicted_class],
            "confidence": confidence,
            "probabilities": {
                "Left": float(probabilities[0]),
                "Centre": float(probabilities[1]),
                "Right": float(probabilities[2])
            },
            "word_importance": word_importance
        }
        
        return result
    
    def _predict_long(self, text: str, get_attributions: bool = False) -> Dict[str, any]:
        """
        Predict using the long model (Longformer) for articles > 512 tokens.
        Longformer can handle up to 4096 tokens.
        
        Args:
            text: Input text
            get_attributions: Not supported for Longformer (ignored)
            
        Returns:
            Prediction dictionary with 5-label mapped to 3-label system
        """
        # Tokenize input (Longformer can handle up to 4096 tokens)
        inputs = self.long_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=4096  # Longformer's max length
        )
        
        # Move inputs to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.long_model(**inputs)
            logits = outputs.logits
            
            # Apply softmax to get probabilities (5 classes)
            probabilities_5 = torch.nn.functional.softmax(logits, dim=-1)
            probabilities_5 = probabilities_5.cpu().numpy()[0]
            
            # Map 5-label probabilities to 3-label system
            probabilities_3 = self._map_5_to_3_labels(probabilities_5)
            
            # Get predicted class and confidence from 3-label system
            probs_array = np.array([
                probabilities_3["Left"],
                probabilities_3["Centre"],
                probabilities_3["Right"]
            ])
            predicted_class = int(np.argmax(probs_array))
            confidence = float(probs_array[predicted_class])
        
        # Prepare result
        result = {
            "label": self.label_map_3[predicted_class],
            "confidence": confidence,
            "probabilities": probabilities_3,
            "word_importance": None,  # Not supported for Longformer
            "raw_5_label_probs": {
                "left": float(probabilities_5[0]),
                "lean left": float(probabilities_5[1]),
                "center": float(probabilities_5[2]),
                "lean right": float(probabilities_5[3]),
                "right": float(probabilities_5[4])
            }
        }
        
        return result
    
    def predict_batch(self, texts: list, max_length: int = 512) -> list:
        """
        Predict bias classifications for multiple texts.
        
        Args:
            texts: List of input article texts
            max_length: Maximum token length for inputs
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        for text in texts:
            result = self.predict(text, max_length)
            results.append(result)
        return results
    
    def _get_word_attributions_short(self, text: str, inputs: dict, max_length: int) -> Dict[str, list]:
        """
        Compute word importance scores for each bias class using SHAP.
        Uses perturbation-based approach to understand contextual importance.
        Only computes for classes with probability > 5% to avoid noise.
        
        Args:
            text: Input text
            inputs: Tokenized inputs
            max_length: Maximum token length
            
        Returns:
            Dictionary with word importance scores for each class
        """
        try:
            import shap
            import numpy as np
            
            # Get predictions first to filter by probability
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits[0], dim=0)
                probs_numpy = probs.cpu().numpy()
            
            # Only compute for classes with probability > 5%
            PROB_THRESHOLD = 0.05
            attributions = {}
            
            # Create a prediction function for SHAP
            def predict_proba(texts):
                """Predict probabilities for a list of texts."""
                if isinstance(texts, str):
                    texts = [texts]
                
                results = []
                for text_item in texts:
                    # Tokenize
                    encoded = self.short_tokenizer(
                        text_item,
                        max_length=max_length,
                        padding='max_length',
                        truncation=True,
                        return_tensors='pt'
                    ).to(self.device)
                    
                    # Predict
                    with torch.no_grad():
                        outputs = self.short_model(**encoded)
                        logits = outputs.logits
                        probs = torch.softmax(logits[0], dim=0).cpu().numpy()
                    
                    results.append(probs)
                
                return np.array(results)
            
            # Use SHAP's Partition explainer (works well for text)
            # Create a simple masker that replaces words with empty string
            masker = shap.maskers.Text(tokenizer=r'\W+')
            
            # Create explainer
            explainer = shap.Explainer(predict_proba, masker, algorithm='partition')
            
            # Get SHAP values
            shap_values = explainer([text])
            
            # Process SHAP values for each class
            for class_idx, class_name in self.label_map_3.items():
                # Skip classes with very low probability
                if probs_numpy[class_idx] < PROB_THRESHOLD:
                    print(f"Skipping attribution for {class_name} (probability: {probs_numpy[class_idx]:.1%} < {PROB_THRESHOLD:.1%})")
                    attributions[class_name] = []
                    continue
                
                # Get SHAP values for this class
                # shap_values.values shape: [1, num_words, num_classes]
                class_shap_values = shap_values.values[0, :, class_idx]
                
                # Get the words
                words = shap_values.data[0]
                
                # Create word-score pairs
                word_scores = []
                for word, score in zip(words, class_shap_values):
                    # Only keep positive contributions (words that increase this class probability)
                    if score > 0 and word.strip():
                        # Filter out common stop words and punctuation
                        if (len(word.strip()) > 0 and 
                            word.lower() not in ['the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'of']):
                            word_scores.append((word.strip(), float(score)))
                
                # Sort by score and take top words
                word_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Normalize scores to 0-1 range
                if word_scores:
                    max_score = max(score for _, score in word_scores)
                    if max_score > 0:
                        word_scores = [(word, score / max_score) for word, score in word_scores]
                
                attributions[class_name] = word_scores
            
            return attributions
            
        except Exception as e:
            print(f"Warning: Could not compute SHAP attributions: {e}")
            import traceback
            traceback.print_exc()
            
            # Fallback to gradient-based method
            print("Falling back to gradient-based attribution...")
            return self._get_word_attributions_gradient(text, inputs, max_length)
    
    def _get_word_attributions_gradient(self, text: str, inputs: dict, max_length: int) -> Dict[str, list]:
        """
        Fallback gradient-based attribution method.
        
        Args:
            text: Input text
            inputs: Tokenized inputs
            max_length: Maximum token length
            
        Returns:
            Dictionary with word importance scores for each class
        """
        try:
            # Re-tokenize to get word mappings
            tokens = self.short_tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            # Get the base model (handles both PEFT and standard models)
            if hasattr(self.short_model, 'base_model'):
                # PEFT model - need to go through base_model.model
                base_model = self.short_model.base_model.model
            else:
                # Standard model
                base_model = self.short_model
            
            # ModernBERT uses 'model.embeddings.tok_embeddings'
            # BERT uses 'bert.embeddings.word_embeddings'
            if hasattr(base_model, 'model'):
                # ModernBERT structure
                embeddings = base_model.model.embeddings.tok_embeddings
                encoder = base_model.model
            elif hasattr(base_model, 'bert'):
                # BERT structure (fallback)
                embeddings = base_model.bert.embeddings.word_embeddings
                encoder = base_model.bert
            else:
                raise AttributeError("Could not find embeddings layer")
            
            # Enable gradients for embeddings
            input_ids = inputs['input_ids'].clone().detach().requires_grad_(False)
            attention_mask = inputs['attention_mask']
            
            # Get embedding vectors
            embed = embeddings(input_ids)
            embed.requires_grad_(True)
            
            # Forward pass with embeddings
            outputs = encoder(
                inputs_embeds=embed,
                attention_mask=attention_mask
            )
            
            # Get sequence output
            sequence_output = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
            
            # Get logits through the full forward pass
            if hasattr(base_model, 'head'):
                # ModernBERT: head -> drop -> classifier
                head_output = base_model.head(sequence_output[:, 0])
                dropped = base_model.drop(head_output)
                logits = base_model.classifier(dropped)
            else:
                # BERT: pooler -> classifier
                pooled_output = outputs[1] if len(outputs) > 1 else sequence_output[:, 0]
                logits = base_model.classifier(pooled_output)
            
            # Get probabilities
            probs = torch.softmax(logits[0], dim=0)
            probs_numpy = probs.detach().cpu().numpy()
            
            # Compute gradients only for classes with probability > 5%
            attributions = {}
            PROB_THRESHOLD = 0.05
            
            for class_idx, class_name in self.label_map_3.items():
                # Skip classes with very low probability
                if probs_numpy[class_idx] < PROB_THRESHOLD:
                    print(f"Skipping attribution for {class_name} (probability: {probs_numpy[class_idx]:.1%} < {PROB_THRESHOLD:.1%})")
                    attributions[class_name] = []
                    continue
                
                # Zero gradients
                if embed.grad is not None:
                    embed.grad.zero_()
                
                # Compute gradient of this class's logit
                class_logit = logits[0, class_idx]
                class_logit.backward(retain_graph=True)
                
                # Get gradient magnitude for each token
                grads = embed.grad[0].abs().sum(dim=-1).cpu().numpy()
                
                # Normalize gradients
                if grads.max() > 0:
                    grads = grads / grads.max()
                
                # Map tokens to words and aggregate scores
                word_scores = self._aggregate_token_scores(tokens, grads)
                attributions[class_name] = word_scores
            
            return attributions
            
        except Exception as e:
            print(f"Warning: Could not compute gradient attributions: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _aggregate_token_scores(self, tokens: list, scores: np.ndarray) -> list:
        """
        Aggregate subword token scores into word-level scores.
        
        Args:
            tokens: List of tokens from tokenizer
            scores: Attribution scores for each token
            
        Returns:
            List of (word, score) tuples
        """
        import string
        
        # Extended punctuation set including quotes and other special chars
        all_punctuation = string.punctuation + '""''–—…'
        
        word_scores = []
        current_word = ""
        current_score = 0
        token_count = 0
        
        for token, score in zip(tokens, scores):
            # Skip special tokens
            if token in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']:
                continue
            
            # Clean the token - remove Ġ (space marker in GPT-style tokenizers)
            # and ## (subword marker in BERT-style tokenizers)
            clean_token = token
            is_new_word = False
            
            if token.startswith('Ġ'):
                # GPT-style tokenization (ModernBERT) - Ġ marks start of new word
                clean_token = token[1:]  # Remove Ġ prefix
                is_new_word = True
            elif token.startswith('##'):
                # BERT-style tokenization - ## marks continuation
                clean_token = token[2:]  # Remove ## prefix
                is_new_word = False
            else:
                # First token or unknown format
                is_new_word = (current_word == "")
            
            # Skip pure punctuation tokens
            if clean_token and all(char in all_punctuation for char in clean_token.strip()):
                continue
            
            # Handle word boundaries
            if is_new_word and current_word:
                # Save previous word
                avg_score = current_score / token_count if token_count > 0 else current_score
                # Only filter out pure punctuation and very common stop words
                if (len(current_word) > 0 and 
                    not all(char in all_punctuation for char in current_word) and
                    current_word.lower() not in ['the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'of']):
                    word_scores.append((current_word, float(avg_score)))
                
                # Start new word
                current_word = clean_token
                current_score = score
                token_count = 1
            else:
                # Continue current word
                current_word += clean_token
                current_score += score
                token_count += 1
        
        # Add last word with same filters
        if current_word:
            avg_score = current_score / token_count if token_count > 0 else current_score
            if (len(current_word) > 0 and 
                not all(char in all_punctuation for char in current_word) and
                current_word.lower() not in ['the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'of']):
                word_scores.append((current_word, float(avg_score)))
        
        return word_scores
    
    def highlight_text_with_attributions(self, text: str, word_importance: Dict[str, list]) -> Dict[str, str]:
        """
        Create HTML highlighted versions of text showing which words contribute to each bias.
        
        Args:
            text: Original text
            word_importance: Dictionary with word importance scores for each class
            
        Returns:
            Dictionary with HTML strings for each class
        """
        if not word_importance:
            return None
        
        import string
        import re
        
        # Extended punctuation set including quotes and other special chars
        all_punctuation = string.punctuation + '""''–—…'
        
        highlighted_versions = {}
        
        # Color intensities for each class
        colors = {
            "Left": "33, 150, 243",      # RGB for blue
            "Centre": "156, 39, 176",     # RGB for purple
            "Right": "244, 67, 54"        # RGB for red
        }
        
        for class_name, word_scores in word_importance.items():
            if not word_scores:
                highlighted_versions[class_name] = text
                continue
            
            # Sort word scores to get top words
            sorted_scores = sorted(word_scores, key=lambda x: x[1], reverse=True)
            
            # Take top 20 words and normalize their scores
            top_words = sorted_scores[:20]
            
            if not top_words:
                highlighted_versions[class_name] = text
                continue
            
            # Normalize scores to 0-1 range based on the top words
            max_score = max(score for _, score in top_words) if top_words else 1.0
            min_score = min(score for _, score in top_words) if top_words else 0.0
            score_range = max_score - min_score if max_score > min_score else 1.0
            
            # Create a mapping of words to normalized scores
            word_score_map = {}
            for word, score in top_words:
                # Normalize to 0-1 range
                normalized_score = (score - min_score) / score_range if score_range > 0 else 0.5
                word_score_map[word.lower()] = normalized_score
            
            # Split text into words while preserving spacing and punctuation
            # This regex captures words, whitespace, and punctuation separately
            tokens = re.findall(r'\w+|[\s\S]', text)
            
            highlighted_text = ""
            for token in tokens:
                token_lower = token.lower().strip()
                
                # Check if this token is a word (not punctuation or whitespace)
                is_word = bool(re.match(r'\w+', token))
                
                if is_word and token_lower in word_score_map:
                    score = word_score_map[token_lower]
                    color = colors[class_name]
                    
                    # Scale opacity: low scores get 0.3, high scores get 0.9
                    opacity = 0.3 + (score * 0.6)
                    
                    # Higher scores get bolder highlighting
                    if score > 0.7:
                        # Very important words - bold with white text
                        style = f'background-color: rgba({color}, {opacity}); color: white; padding: 2px 4px; border-radius: 3px; font-weight: bold;'
                    elif score > 0.4:
                        # Moderately important - colored background
                        style = f'background-color: rgba({color}, {opacity}); padding: 2px 4px; border-radius: 3px;'
                    else:
                        # Less important - subtle highlight
                        style = f'background-color: rgba({color}, {opacity * 0.6}); padding: 1px 2px; border-radius: 2px;'
                    
                    highlighted_text += f'<mark style="{style}">{token}</mark>'
                else:
                    # Not a word to highlight - just add it as is
                    highlighted_text += token
            
            highlighted_versions[class_name] = highlighted_text
        
        return highlighted_versions
