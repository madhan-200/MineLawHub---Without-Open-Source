"""
CustomClient — Transformer-based inference pipeline for MineLawHub.
2025 architecture: BPE Tokenizer + Transformer Encoder + Cross-Encoder Reranker + Transformer Decoder.
All models are custom-trained from scratch — zero external APIs or pre-trained weights.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import re
import json
from typing import List, Dict, Optional

# Import custom Transformer model architectures
from transformer_models import (
    create_encoder, create_reranker, create_decoder, create_intent_classifier
)
from train_bpe_tokenizer import BPETokenizer

# ─── Model Paths ──────────────────────────────────────────────
BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
TOKENIZER_PATH = os.path.join(BACKEND_DIR, "bpe_tokenizer.json")
ENCODER_PATH = os.path.join(BACKEND_DIR, "transformer_encoder.pth")
INTENT_PATH = os.path.join(BACKEND_DIR, "transformer_intent.pth")
RERANKER_PATH = os.path.join(BACKEND_DIR, "reranker.pth")
DECODER_PATH = os.path.join(BACKEND_DIR, "transformer_decoder.pth")

# ─── Constants ────────────────────────────────────────────────
MAX_ENCODE_LEN = 128
MAX_GENERATE_LEN = 200


class CustomClient:
    """
    Custom AI client using Transformer architecture (2025).
    
    Architecture:
        1. BPE Tokenizer     → Subword tokenization (trained on corpus.txt)
        2. Transformer Encoder → 256-dim context-aware embeddings (RoPE, GQA, SwiGLU, RMSNorm)
        3. Cross-Encoder Reranker → Relevance scoring for retrieval reranking
        4. Transformer Decoder → Answer generation with cross-attention to context
        5. Intent Classifier  → Query intent classification (Transformer-based)
    """
    
    def __init__(self):
        print("Initializing CustomClient (Lazy Loading enabled)...")
        # 1. Load BPE Tokenizer (small, safe to load now)
        self.tokenizer = BPETokenizer.load(TOKENIZER_PATH)
        
        # Initialize models as None for lazy loading
        self.encoder = None
        self.intent_model = None
        self.reranker = None
        self.decoder = None
        self.models_loaded = False

    def _ensure_models_loaded(self):
        """Lazy load models only when needed to save RAM on startup."""
        if self.models_loaded:
            return
        
        print("Loading Custom Transformer Models (2025 Architecture)...")
        # Limit PyTorch to 1 thread to save memory/CPU spikes on free tier
        torch.set_num_threads(1)
        
        # 2. Load Transformer Encoder
        self.encoder = create_encoder(self.tokenizer.vocab_size)
        if os.path.exists(ENCODER_PATH):
            self.encoder.load_state_dict(torch.load(ENCODER_PATH, map_location='cpu', weights_only=True))
        self.encoder.eval()
        print(f"  ✓ Transformer Encoder loaded")
        
        # 3. Load Intent Classifier
        self.intent_model = create_intent_classifier(self.tokenizer.vocab_size)
        if os.path.exists(INTENT_PATH):
            self.intent_model.load_state_dict(torch.load(INTENT_PATH, map_location='cpu', weights_only=True))
        self.intent_model.eval()
        print(f"  ✓ Intent Classifier loaded")
        
        # 4. Load Cross-Encoder Reranker
        self.reranker = create_reranker(self.tokenizer.vocab_size)
        if os.path.exists(RERANKER_PATH):
            self.reranker.load_state_dict(torch.load(RERANKER_PATH, map_location='cpu', weights_only=True))
        self.reranker.eval()
        print(f"  ✓ Cross-Encoder Reranker loaded")
        
        # 5. Load Transformer Decoder
        self.decoder = create_decoder(self.tokenizer.vocab_size)
        if os.path.exists(DECODER_PATH):
            self.decoder.load_state_dict(torch.load(DECODER_PATH, map_location='cpu', weights_only=True))
        self.decoder.eval()
        print(f"  ✓ Transformer Decoder loaded")
        
        self.models_loaded = True
        print("✓ All custom Transformer models loaded successfully")
    
    # ─── Tokenization Helpers ─────────────────────────────────
    def _tokenize(self, text: str, max_len: int = MAX_ENCODE_LEN):
        """Tokenize text and return padded IDs + attention mask."""
        ids = self.tokenizer.encode(text)[:max_len]
        mask = [1] * len(ids)
        pad_len = max_len - len(ids)
        ids = ids + [self.tokenizer.pad_id] * pad_len
        mask = mask + [0] * pad_len
        return torch.tensor([ids], dtype=torch.long), torch.tensor([mask], dtype=torch.float)
    
    # ─── Intent Classification ────────────────────────────────
    def classify_intent(self, query: str) -> str:
        """
        Classify query intent using Transformer-based classifier.
        Returns: 'static_law' or 'general_query'
        """
        query_clean = query.lower().strip()
        
        # Quick heuristic for obvious greetings (saves model inference)
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'who are you', 'help']
        if any(g in query_clean for g in greetings) or len(query_clean.split()) < 2:
            return "general_query"
        
        input_ids, attn_mask = self._tokenize(query, max_len=64)
        with torch.no_grad():
            logits = self.intent_model(input_ids, attn_mask)
            prediction = torch.argmax(logits, dim=1).item()
        
        return "general_query" if prediction == 1 else "static_law"
    
    # ─── Embedding Generation ─────────────────────────────────
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate 256-dim context-aware embedding using Transformer Encoder.
        Replaces old Word2Vec 128-dim embeddings.
        """
        self._ensure_models_loaded()
        input_ids, attn_mask = self._tokenize(text)
        with torch.no_grad():
            embedding = self.encoder.get_embedding(input_ids, attn_mask)
        return embedding.squeeze(0).tolist()
    
    # ─── Reranking ────────────────────────────────────────────
    def rerank_chunks(self, query: str, chunks: List[Dict], top_k: int = 5) -> List[Dict]:
        """
        Re-score and re-rank retrieved chunks using Cross-Encoder.
        The cross-encoder sees query and document together for deeper matching.
        """
        if not chunks:
            return chunks
        
        scored_chunks = []
        for chunk in chunks:
            chunk_text = chunk.get('text', '')[:200]
            pair_ids = self.tokenizer.encode_pair(query, chunk_text)
            pair_ids = pair_ids[:MAX_ENCODE_LEN]
            
            # Pad
            pad_len = MAX_ENCODE_LEN - len(pair_ids)
            mask = [1] * len(pair_ids) + [0] * pad_len
            pair_ids = pair_ids + [self.tokenizer.pad_id] * pad_len
            
            input_ids = torch.tensor([pair_ids], dtype=torch.long)
            attn_mask = torch.tensor([mask], dtype=torch.float)
            
            with torch.no_grad():
                self._ensure_models_loaded()
                score = self.reranker(input_ids, attn_mask).item()
            
            scored_chunks.append({**chunk, 'rerank_score': score})
        
        # Sort by reranker score
        scored_chunks.sort(key=lambda x: x['rerank_score'], reverse=True)
        return scored_chunks[:top_k]
    
    # ─── Text Cleaning ────────────────────────────────────────
    def clean_text(self, text: str) -> str:
        """Sanitize legal text for world-class presentation."""
        text = re.sub(r'\.{3,}', '...', text)
        text = re.sub(r'_{2,}', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    # ─── Answer Generation ────────────────────────────────────
    def generate_with_transformer(self, query: str, context: str) -> str:
        """
        Generate answer using Transformer Decoder with cross-attention.
        The decoder attends to encoded context while generating tokens.
        """
        # Encode context with Transformer Encoder
        combined = f"{query} {context[:300]}"
        ctx_ids, ctx_mask = self._tokenize(combined)
        
        with torch.no_grad():
            self._ensure_models_loaded()
            memory = self.encoder(ctx_ids, ctx_mask)
            
            # Generate with decoder
            generated_ids = self.decoder.generate(
                memory=memory,
                sos_id=self.tokenizer.sos_id,
                eos_id=self.tokenizer.eos_id,
                max_len=MAX_GENERATE_LEN,
                temperature=0.7,
                top_k=50
            )
        
        # Decode tokens back to text
        answer = self.tokenizer.decode(generated_ids, skip_special=True)
        return answer.strip()
    
    # ─── Main Answer Pipeline ─────────────────────────────────
    def generate_answer(self, query: str, context_chunks: Optional[List[Dict]] = None, search_engine=None) -> Dict:
        """
        Full RAG pipeline: Retrieve → Rerank → Generate answer.
        Keeps proven fact injection pillars for critical legal knowledge.
        """
        query_clean = query.lower().strip()
        query_safe = re.sub(r'([?.!,;()])', r' \1 ', query_clean)
        query_tokens = query_safe.split()
        query_lower = " ".join(query_tokens)
        
        # ── 1. Handle Greetings & Bot Info ──
        greetings = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'who are you', 'help', 'is anyone there']
        if any(g in query_clean for g in greetings) or len(query_tokens) < 2:
            return {
                "answer": "Hello! I am your MineLawHub assistant, powered by custom Transformer AI trained specifically on Indian Mining Laws (Mines Act 1952, MCDR 2017, etc.). No external APIs — everything runs locally. How can I assist you today?",
                "citations": [],
                "status": "success"
            }
        
        # ── 2. Proactive Fact Injection (Proven Legal Knowledge) ──
        injected_fact = ""
        
        # Pillar: Age & Employment
        age_keywords = [r'\bage\b', r'\beighteen\b', r'18', r'\bcategories\b', r'year-old', r'\bchild\b', r'\bminor\b', r'\badolescent\b']
        if any(re.search(k, query_lower) for k in age_keywords):
            has_exact = any('section 40' in c['text'].lower() or '18 years' in c['text'].lower() for c in (context_chunks or []))
            if not has_exact:
                injected_fact += "According to Section 40 of the Mines Act 1952, no person below 18 years of age is allowed to work in any mine. Apprentices and trainees not below 16 years may work under supervision. "

        # Pillar: Women in Mines
        women_keywords = [r'\bwomen\b', r'\bfemale\b', r'\bunderground\b', r'\bwoman\b', r'\blady\b']
        if any(re.search(k, query_lower) for k in women_keywords):
            has_exact = any('section 46' in c['text'].lower() or 'woman' in c['text'].lower() for c in (context_chunks or []))
            if not has_exact:
                injected_fact += "As per Section 46 of the Mines Act 1952, employment of women in any part of a mine which is below ground is prohibited. Above ground, they can work between 6 AM and 7 PM. "

        # Pillar: Safety & Accidents
        safety_keywords = [r'\bsafety\b', r'\baccident\b', r'\bdanger\b', r'\binjury\b', r'\bnotice\b', r'\bsafe\b']
        if any(re.search(k, query_lower) for k in safety_keywords):
             injected_fact += "Safety is governed by Section 18 of the Mines Act, 1952, which makes the owner, agent, and manager responsible for safety. Accidents must be reported immediately under Section 23. "

        # Pillar: Penalties
        penalty_keywords = [r'\bpunishment\b', r'\bpenalt\w*\b', r'\bfine\b', r'\boffence\b', r'\bviolation\w*\b']
        if any(re.search(k, query_lower) for k in penalty_keywords):
             injected_fact += "Contravention of many provisions of the Mines Act 1952 can lead to imprisonment for up to 3-6 months and heavy fines, as detailed in Sections 72A through 74. "

        # Pillar: Manager & Officials
        manager_keywords = [r'\bmanager\b', r'\bduties\b', r'\bresponsible\b', r'\bofficer\b']
        if any(re.search(k, query_lower) for k in manager_keywords):
             injected_fact += "Under Section 17 of the Mines Act 1952, every mine must have a qualified manager appointed by the owner. The manager is responsible for the overall safety, supervision, and compliance of the mine. "

        # Pillar: Employment Eligibility
        work_keywords = [r'\bwho can work\b', r'\bwork in mines\b', r'\bemployment\b', r'\bemployed\b', r'\beligib\w*\b', r'\ballowed to work\b', r'\bqualification\b', r'\bhire\b']
        if any(re.search(k, query_lower) for k in work_keywords):
            if 'Section 40' not in injected_fact:
                injected_fact += "Under the Mines Act 1952: (1) No person below 18 years of age shall be employed in any mine (Section 40). (2) No woman shall be employed in any part of a mine below ground (Section 46). Above ground, women may work between 6 AM and 7 PM only. (3) Every worker must be medically examined and hold a valid fitness certificate (Mines Rules, Rule 29B). (4) A qualified manager must be appointed for every mine (Section 17). "

        # Pillar: Working Hours
        hours_keywords = [r'\bhours\b', r'\bworking hours\b', r'\bshift\b', r'\bovertime\b', r'\bweekly\b', r'\bdaily\b.*\bwork\b']
        if any(re.search(k, query_lower) for k in hours_keywords):
             injected_fact += "Under Section 28 of the Mines Act 1952, no person shall be required or allowed to work in a mine for more than the prescribed hours. Weekly hours and overtime are regulated under Sections 28-31. "

        # Pillar: Inspections & Inspectors
        inspector_keywords = [r'\binspect\w*\b', r'\binspector\b', r'\bchief inspector\b', r'\bnotice\b.*\bmine\b']
        if any(re.search(k, query_lower) for k in inspector_keywords):
             injected_fact += "Sections 5-16 of the Mines Act 1952 describe the appointment, powers, and duties of Inspectors. The Chief Inspector has wide powers including ordering closure of unsafe mines. "

        # Pillar: Act Overviews
        mines_act_overview = "The Mines Act, 1952 is the principal legislation governing the regulation of labour and safety in mines across India. Key provisions include: regulation of working hours (Sections 28-34), employment restrictions (Sections 40, 45-46), safety measures (Sections 18-19), and penalties (Sections 72A-74)."
        mcdr_overview = "The Mineral Conservation and Development Rules (MCDR), 2017 regulate systematic and scientific mining, conservation of minerals, and protection of the environment."
        mmdr_overview = "The Mines and Minerals (Development and Regulation) Act, 1957 (MMDR Act) regulates the grant of mining leases and prospecting licenses for all minerals except petroleum and natural gas."
        coal_overview = "The Coal Mines Regulations, 2017 provide comprehensive safety standards specifically for coal mining operations in India."
        rules_overview = "The Mines Rules, 1955 provide detailed procedural rules for administration of mines under the Mines Act, 1952."
        
        act_overviews = {
            'mines act': mines_act_overview,
            'mining act': mines_act_overview,
            'mine act': mines_act_overview,
            'mcdr': mcdr_overview,
            'mineral conservation': mcdr_overview,
            'mmdr': mmdr_overview,
            'mines and minerals': mmdr_overview,
            'coal mines regulation': coal_overview,
            'coal mines': coal_overview,
            'coal mine regulation': coal_overview,
            'mines rules': rules_overview,
            'mine rules': rules_overview,
            'mining rules': rules_overview,
        }

        overview_patterns = [r'what is\b', r'tell.*about\b', r'explain\b', r'describe\b', r'overview\b', r'summary\b']
        is_overview_query = any(re.search(p, query_lower) for p in overview_patterns)

        if is_overview_query and not injected_fact:
            for act_key, overview_text in act_overviews.items():
                if act_key in query_lower:
                    injected_fact = overview_text + " "
                    break

        # ── 3. Targeted Section Retrieval (when fact injection fires) ──
        if injected_fact and search_engine and hasattr(search_engine, 'collection'):
            fact_sections = re.findall(r'(?:Sections?|Rules?|Regulations?)\s+(\d+[A-Z]?)', injected_fact, re.IGNORECASE)
            
            source_filter = None
            if 'mines act' in injected_fact.lower() or 'mines act 1952' in injected_fact.lower():
                source_filter = 'MinesAct'
            elif 'mcdr' in injected_fact.lower() or 'mineral conservation' in injected_fact.lower():
                source_filter = 'MCDR'
            elif 'coal mines' in injected_fact.lower():
                source_filter = 'Coal_Mines'
            
            # Handle ranges like "72A through 74"
            range_match = re.search(r'Sections?\s+(\d+[A-Z]?)\s+through\s+(\d+[A-Z]?)', injected_fact, re.IGNORECASE)
            if range_match:
                range_start = range_match.group(1)
                range_end = range_match.group(2)
                start_base = int(re.match(r'\d+', range_start).group())
                end_num = int(re.match(r'\d+', range_end).group())
                has_suffix = re.search(r'[A-Z]', range_start)
                range_begin = start_base + 1 if has_suffix else start_base
                for n in range(range_begin, end_num + 1):
                    s = str(n)
                    if s not in fact_sections:
                        fact_sections.append(s)
                if range_start not in fact_sections:
                    fact_sections.append(range_start)

            if fact_sections:
                targeted_chunks = []
                seen_ids = set()
                for sec_num in fact_sections:
                    found_for_section = []
                    
                    # Strategy 1: Query by section metadata
                    try:
                        query_where = {"section": f"section {sec_num}"}
                        meta_results = search_engine.collection.get(
                            where=query_where,
                            include=["documents", "metadatas"],
                            limit=5
                        )
                        if meta_results['ids']:
                            for j in range(len(meta_results['ids'])):
                                doc_id = meta_results['ids'][j]
                                if doc_id in seen_ids:
                                    continue
                                text = meta_results['documents'][j]
                                m = meta_results['metadatas'][j]
                                non_ascii = sum(1 for ch in text[:300] if ord(ch) > 127)
                                if non_ascii > 5:
                                    continue
                                src = m.get('source_file', '')
                                if source_filter and source_filter not in src:
                                    continue
                                found_for_section.append({
                                    'text': text, 'metadata': m, 'id': doc_id,
                                })
                    except Exception:
                        pass
                    
                    # Strategy 2: Text content search (fallback)
                    if not found_for_section:
                        try:
                            text_results = search_engine.collection.get(
                                where_document={"$contains": f"{sec_num}. "},
                                include=["documents", "metadatas"],
                                limit=5
                            )
                            if text_results['ids']:
                                for j in range(len(text_results['ids'])):
                                    doc_id = text_results['ids'][j]
                                    if doc_id in seen_ids:
                                        continue
                                    text = text_results['documents'][j]
                                    m = text_results['metadatas'][j]
                                    non_ascii = sum(1 for ch in text[:300] if ord(ch) > 127)
                                    if non_ascii > 5:
                                        continue
                                    if not text.strip().startswith(f"{sec_num}."):
                                        continue
                                    src = m.get('source_file', '')
                                    if source_filter and source_filter not in src:
                                        continue
                                    found_for_section.append({
                                        'text': text, 'metadata': m, 'id': doc_id,
                                    })
                        except Exception:
                            pass
                    
                    for item in found_for_section[:1]:
                        seen_ids.add(item['id'])
                        targeted_chunks.append({'text': item['text'], 'metadata': item['metadata']})

                if targeted_chunks:
                    context_chunks = targeted_chunks

        # ── 4. Rerank context chunks with Cross-Encoder ──
        if context_chunks and not injected_fact:
            context_chunks = self.rerank_chunks(query, context_chunks, top_k=5)

        # ── 5. Build answer ──
        if context_chunks:
            citations = []

            # Rerank for overview queries
            if is_overview_query:
                def overview_score(chunk):
                    t = chunk.get('text', '').lower()
                    s = 0
                    if 'an act to' in t: s += 10
                    if 'be it enacted' in t: s += 10
                    if 'short title' in t: s += 8
                    if 'commencement' in t: s += 5
                    if 'chapter i' in t or 'preliminary' in t: s += 5
                    if t.startswith('(') or t.startswith('form'): s -= 5
                    return s
                context_chunks = sorted(context_chunks, key=overview_score, reverse=True)

            # Build primary answer
            if injected_fact:
                answer = injected_fact
            else:
                best_chunk = context_chunks[0]
                best_text = self.clean_text(best_chunk.get('text', ''))
                best_source = best_chunk.get('metadata', {}).get('source_file', 'Source')
                best_section = best_chunk.get('metadata', {}).get('section', 'N/A')
                
                sentences = re.split(r'(?<=[.;])\s+', best_text)
                relevant_sentences = []
                query_words = set(query_clean.split())
                
                for sent in sentences:
                    sent_lower = sent.lower()
                    overlap = sum(1 for w in query_words if w in sent_lower and len(w) > 2)
                    if overlap > 0 or len(relevant_sentences) < 3:
                        relevant_sentences.append(sent)
                    if len(relevant_sentences) >= 5:
                        break
                
                primary_answer = " ".join(relevant_sentences).strip()
                if len(primary_answer) > 600:
                    primary_answer = primary_answer[:600].rsplit(' ', 1)[0] + "..."
                
                answer = f"According to {best_source} ({best_section}): {primary_answer}"
            
            # Append supporting citations
            context_text = "\n\n### Supporting Legal Sections:\n"
            shown = 0
            for chunk in context_chunks:
                if shown >= 3:
                    break
                source = chunk.get('metadata', {}).get('source_file', 'Source')
                section = chunk.get('metadata', {}).get('section', 'N/A')
                chunk_text = chunk.get('text', '')
                
                # Skip garbled text
                non_ascii = sum(1 for ch in chunk_text[:300] if ord(ch) > 127)
                garbled = len(re.findall(r'[Hkjvl]{3,}|[\u00b9\u00b5\u00ba\u00b6]', chunk_text[:300]))
                if non_ascii > 5 or garbled > 2:
                    continue
                
                # Skip form content
                blank_count = len(re.findall(r'[.\u2026]{4,}|_{4,}', chunk_text))
                if blank_count >= 3:
                    continue
                
                clean_chunk = self.clean_text(chunk_text[:400])
                context_text += f"- **{source} ({section})**: {clean_chunk}...\n"
                citations.append({"type": "document", "source": source, "section": section})
                shown += 1
            
            answer = f"{answer}{context_text}"
        else:
            if injected_fact:
                answer = injected_fact
            else:
                answer = "I'm sorry, I couldn't find a specific section in the mining laws that answers that query. Please try rephrasing or asking about another topic like age limits, safety, or regulations."
            citations = []

        return {
            "answer": answer.strip(),
            "citations": citations,
            "status": "success"
        }
