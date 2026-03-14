"""
ChromaDB search engine for semantic retrieval.
Handles vector similarity search for mining law documents.
Uses Hybrid Search: Semantic + Lexical + Source-Aware boosting.
"""

import os
import re
from pathlib import Path
from typing import List, Dict, Optional
import chromadb
from custom_client import CustomClient


class SearchEngine:
    """
    Encapsulates ChromaDB retrieval logic for hybrid search.
    """
    
    def __init__(self, chroma_path: Optional[str] = None, custom_client=None):
        """
        Initialize the search engine.
        
        Args:
            chroma_path: Path to ChromaDB storage (optional)
            custom_client: Shared CustomClient instance (avoids doubling memory)
        """
        if chroma_path is None:
            script_dir = Path(__file__).parent.parent
            chroma_path = str(script_dir / "embeddings" / "chroma_store_v3")
            # If v3 doesn't exist, try to auto-rebuild it
            if not os.path.exists(chroma_path):
                print("ChromaDB store not found. Auto-rebuilding...")
                self._auto_rebuild_chromadb(script_dir, chroma_path)
            # Fallback to v2 if v3 still doesn't exist
            if not os.path.exists(chroma_path):
                chroma_path = str(script_dir / "embeddings" / "chroma_store_v2")
        
        print(f"Connecting to ChromaDB at {chroma_path}")
        
        self.chroma_path = chroma_path
        # Use shared client if provided, otherwise create new (for standalone usage)
        self.custom_client = custom_client if custom_client is not None else CustomClient()
        self.client = None
        self.collection = None
        
        # Source mapping: common query phrases -> ChromaDB source_file values
        self.source_map = {
            'mines act': 'MinesAct1952',
            'mines act 1952': 'MinesAct1952',
            'mcdr': 'MCDR_2017',
            'mcdr 2017': 'MCDR_2017',
            'mineral conservation': 'MCDR_2017',
            'coal mines regulation': 'Coal_Mines_Regulation_2017_Noti',
            'coal regulation': 'Coal_Mines_Regulation_2017_Noti',
            'mine regulations 1961': 'MineRegulations1961_13092023',
            'metalliferous': 'MineRegulations1961_13092023',
            'mines rules': 'Mines_Rules_1955',
            'mines rules 1955': 'Mines_Rules_1955',
            'mmdr': 'mmdr_act,1957',
            'mmdr act': 'mmdr_act,1957',
            'mineral development': 'mmdr_act,1957',
        }
        
        # Stop words to ignore during lexical matching
        self.stop_words = {
            'what', 'the', 'is', 'are', 'was', 'were', 'how', 'does',
            'can', 'tell', 'about', 'explain', 'describe', 'which', 'who',
            'and', 'for', 'this', 'that', 'with', 'from', 'have', 'has',
            'been', 'will', 'would', 'could', 'should', 'may', 'might',
            'shall', 'not', 'but', 'all', 'any', 'each', 'every',
        }
        
        self._initialize_client()
    
    def _auto_rebuild_chromadb(self, project_root: Path, target_path: str):
        """Auto-rebuild ChromaDB embeddings from source text files."""
        try:
            import subprocess
            import sys
            rebuild_script = project_root / "embeddings" / "rebuild_transformer_embeddings.py"
            if rebuild_script.exists():
                print(f"Running {rebuild_script}...")
                subprocess.run(
                    [sys.executable, str(rebuild_script)],
                    cwd=str(project_root),
                    check=True,
                    timeout=600
                )
                print("✓ ChromaDB rebuild complete!")
            else:
                print(f"✗ Rebuild script not found at {rebuild_script}")
        except Exception as e:
            print(f"✗ Auto-rebuild failed: {e}")
    
    def _initialize_client(self):
        """Initialize ChromaDB client and collection."""
        try:
            self.client = chromadb.PersistentClient(path=self.chroma_path)
            self.collection = self.client.get_collection(name="mining_law_docs")
            print(f"✓ Connected to ChromaDB at {self.chroma_path}")
            print(f"✓ Collection contains {self.collection.count()} documents")
        except Exception as e:
            print(f"✗ Error initializing ChromaDB: {str(e)}")
            print("Please run embeddings/rebuild_transformer_embeddings.py first.")
            raise
    
    def _detect_target_source(self, query_lower: str) -> Optional[str]:
        """Detect if query references a specific Act/Rule document. Supports fuzzy matching for typos."""
        # Exact match first (sorted by length desc for longest match)
        for phrase in sorted(self.source_map.keys(), key=len, reverse=True):
            if phrase in query_lower:
                return self.source_map[phrase]
        
        # Fuzzy match: check if any phrase has high character overlap with query
        query_words = set(query_lower.split())
        best_match = None
        best_score = 0.0
        
        for phrase, source in self.source_map.items():
            phrase_words = set(phrase.split())
            # For each phrase word, find best matching query word by char overlap
            word_scores = []
            for pw in phrase_words:
                best_word_score = 0.0
                for qw in query_words:
                    # Character bigram overlap
                    pw_bigrams = set(pw[i:i+2] for i in range(len(pw)-1)) if len(pw) > 1 else {pw}
                    qw_bigrams = set(qw[i:i+2] for i in range(len(qw)-1)) if len(qw) > 1 else {qw}
                    if pw_bigrams or qw_bigrams:
                        overlap = len(pw_bigrams & qw_bigrams) / max(len(pw_bigrams | qw_bigrams), 1)
                        best_word_score = max(best_word_score, overlap)
                word_scores.append(best_word_score)
            
            if word_scores:
                avg_score = sum(word_scores) / len(word_scores)
                if avg_score > best_score and avg_score > 0.5:  # 50% threshold
                    best_score = avg_score
                    best_match = source
        
        return best_match
    
    def _get_important_tokens(self, query: str) -> List[str]:
        """Extract important query terms (remove stop words)."""
        tokens = [t.lower() for t in query.split() if len(t) > 2]
        return [t for t in tokens if t not in self.stop_words]
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Perform robust hybrid search:
        1. Semantic (ChromaDB vector similarity)
        2. Lexical (exact keyword matching via ChromaDB where_document)
        3. Source-Aware boosting (if query references a specific Act)
        """
        try:
            query_lower = query.lower()
            query_embedding = self.custom_client.get_embedding(query)
            target_source = self._detect_target_source(query_lower)
            important_tokens = self._get_important_tokens(query)
            
            candidates = {}  # doc_id -> candidate dict (deduplication)
            
            # --- STAGE 1: Semantic Retrieval (broad pool) ---
            semantic_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k * 6, self.collection.count()),
                include=['documents', 'metadatas', 'distances']
            )
            
            if semantic_results['ids'] and len(semantic_results['ids']) > 0:
                for i in range(len(semantic_results['ids'][0])):
                    doc_id = semantic_results['ids'][0][i]
                    candidates[doc_id] = {
                        'text': semantic_results['documents'][0][i],
                        'metadata': semantic_results['metadatas'][0][i],
                        'semantic_score': 1.0 / (1.0 + semantic_results['distances'][0][i]),
                        'lexical_score': 0.0,
                        'source_boost': 0.0
                    }
            
            # --- STAGE 2: Lexical Retrieval (keyword filter via ChromaDB) ---
            if important_tokens:
                for token in important_tokens[:3]:
                    try:
                        lexical_results = self.collection.get(
                            where_document={"$contains": token},
                            include=['documents', 'metadatas'],
                            limit=top_k * 4
                        )
                        if lexical_results['ids']:
                            for j in range(len(lexical_results['ids'])):
                                doc_id = lexical_results['ids'][j]
                                if doc_id not in candidates:
                                    candidates[doc_id] = {
                                        'text': lexical_results['documents'][j],
                                        'metadata': lexical_results['metadatas'][j],
                                        'semantic_score': 0.0,
                                        'lexical_score': 0.0,
                                        'source_boost': 0.0
                                    }
                    except Exception:
                        pass
            
            # --- STAGE 2b: Source-filtered retrieval ---
            if target_source:
                try:
                    source_results = self.collection.get(
                        where={"source_file": target_source},
                        include=['documents', 'metadatas'],
                        limit=top_k * 4
                    )
                    if source_results['ids']:
                        for j in range(len(source_results['ids'])):
                            doc_id = source_results['ids'][j]
                            if doc_id not in candidates:
                                candidates[doc_id] = {
                                    'text': source_results['documents'][j],
                                    'metadata': source_results['metadatas'][j],
                                    'semantic_score': 0.0,
                                    'lexical_score': 0.0,
                                    'source_boost': 0.0
                                }
                except Exception:
                    pass
            
            # --- STAGE 2c: Section-Keyword Expansion ---
            # For common query patterns, inject targeted legal phrases
            import re
            section_keywords = []
            if re.search(r'\b(work|employ\w*|age|eighteen|child|minor|who can)\b', query_lower):
                section_keywords.extend(['Employment of persons', 'eighteen years', 'Employment of women'])
            if re.search(r'\b(women|woman|female|underground)\b', query_lower):
                section_keywords.append('Employment of women')
            if re.search(r'\b(safety|accident\w*|danger\w*|injur\w*)\b', query_lower):
                section_keywords.extend(['Notice of accidents', 'causes of danger', 'safety of persons'])
            if re.search(r'\b(penalt\w*|fine\w*|offence\w*|violation\w*|punish\w*|imprison\w*)\b', query_lower):
                section_keywords.extend(['Enhanced penalty', 'contravention', 'shall be punishable', 'imprisonment'])
            if re.search(r'\b(hours|shift|overtime|weekly)\b', query_lower):
                section_keywords.extend(['Hours of work', 'Extra wages for overtime'])
            if re.search(r'\b(inspect\w*|examin\w*)\b', query_lower):
                section_keywords.extend(['Powers of Inspectors', 'Medical appliance'])
            # Section-specific queries (e.g., "What is Section 40?")
            sec_match = re.search(r'\b(?:section|rule|regulation)\s+(\d+[A-Z]?)\b', query_lower)
            if sec_match:
                sec_num = sec_match.group(1)
                section_keywords.append(f'{sec_num}.')
            
            keyword_expansion_ids = set()
            for phrase in section_keywords:
                try:
                    phrase_results = self.collection.get(
                        where_document={"$contains": phrase},
                        include=['documents', 'metadatas'],
                        limit=5
                    )
                    if phrase_results['ids']:
                        for j in range(len(phrase_results['ids'])):
                            doc_id = phrase_results['ids'][j]
                            keyword_expansion_ids.add(doc_id)
                            if doc_id not in candidates:
                                candidates[doc_id] = {
                                    'text': phrase_results['documents'][j],
                                    'metadata': phrase_results['metadatas'][j],
                                    'semantic_score': 0.0,
                                    'lexical_score': 0.0,
                                    'source_boost': 0.0
                                }
                except Exception:
                    pass
            
            # --- STAGE 3: Score all candidates ---
            for doc_id, c in candidates.items():
                text_lower = c['text'].lower()
                text_raw = c['text']
                source = c['metadata'].get('source_file', '')
                
                # Lexical score: fraction of important query tokens found in text
                if important_tokens:
                    matches = sum(1 for t in important_tokens if t in text_lower)
                    c['lexical_score'] = matches / len(important_tokens)
                
                # Source boost: if query specifically references this Act/Rule
                if target_source and source == target_source:
                    c['source_boost'] = 1.0
                
                # --- FORM-CONTENT PENALTY ---
                # Penalize chunks that look like fill-in-the-blank forms
                import re
                blank_count = len(re.findall(r'[.\u2026]{4,}|_{4,}', text_raw))
                form_penalty = 1.0
                if blank_count >= 3:
                    form_penalty = 0.2  # Heavy penalty for form content
                elif blank_count >= 1:
                    form_penalty = 0.5  # Moderate penalty
                
                # --- GARBLED/NON-ENGLISH TEXT PENALTY ---
                # Penalize chunks with non-ASCII or garbled text patterns
                non_ascii = sum(1 for ch in text_raw[:300] if ord(ch) > 127)
                # Also check for garbled Hindi-encoded patterns (consecutive consonants)
                import re as re2
                garbled = len(re2.findall(r'[Hkjvl]{3,}|[\u00b9\u00b5\u00ba\u00b6]', text_raw[:300]))
                if non_ascii > 5 or garbled > 2:
                    form_penalty *= 0.05  # Near-zero for garbled text
                
                # --- FINAL HYBRID SCORE ---
                # Keyword expansion boost for directly matched section chunks
                kw_boost = 0.8 if doc_id in keyword_expansion_ids else 0.0
                
                if target_source:
                    c['final_score'] = (
                        0.25 * c['semantic_score'] +
                        0.35 * c['lexical_score'] +
                        0.40 * c['source_boost'] +
                        kw_boost
                    ) * form_penalty
                else:
                    c['final_score'] = (
                        0.50 * c['semantic_score'] +
                        0.50 * c['lexical_score'] +
                        kw_boost
                    ) * form_penalty
            
            # Sort by final hybrid score
            ranked = sorted(candidates.values(), key=lambda x: x['final_score'], reverse=True)
            
            return [
                {'text': c['text'], 'metadata': c['metadata'], 'score': c['final_score']}
                for c in ranked[:top_k]
            ]
            
        except Exception as e:
            print(f"✗ Search error: {str(e)}")
            return []
    
    def get_collection_stats(self) -> Dict:
        """
        Get statistics about the document collection.
        
        Returns:
            Dictionary with collection statistics
        """
        try:
            count = self.collection.count()
            return {
                'total_documents': count,
                'collection_name': 'mining_law_docs',
                'status': 'ready'
            }
        except Exception as e:
            return {
                'total_documents': 0,
                'collection_name': 'mining_law_docs',
                'status': 'error',
                'error': str(e)
            }
