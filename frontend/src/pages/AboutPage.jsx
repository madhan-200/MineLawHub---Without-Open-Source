import React from 'react';
import { Box, Typography, Paper, Grid, Chip } from '@mui/material';
import { motion } from 'framer-motion';
import PsychologyIcon from '@mui/icons-material/Psychology';
import SearchIcon from '@mui/icons-material/Search';
import QuestionAnswerIcon from '@mui/icons-material/QuestionAnswer';
import VerifiedIcon from '@mui/icons-material/Verified';
import SecurityIcon from '@mui/icons-material/Security';
import MemoryIcon from '@mui/icons-material/Memory';
import AnimatedAvatar from '../components/AnimatedAvatar';

const steps = [
    {
        icon: <QuestionAnswerIcon sx={{ fontSize: 36 }} />,
        title: 'Step 1: You Ask',
        desc: 'Type your question about Indian mining law. Text is tokenized using our custom BPE tokenizer (8,000 subword vocabulary).',
        color: '#1565c0',
    },
    {
        icon: <PsychologyIcon sx={{ fontSize: 36 }} />,
        title: 'Step 2: Intent Classification',
        desc: 'Our Transformer Intent Classifier determines if this is a legal query or a greeting — 100% accuracy.',
        color: '#7b1fa2',
    },
    {
        icon: <SearchIcon sx={{ fontSize: 36 }} />,
        title: 'Step 3: Hybrid Search',
        desc: '256-dim Transformer embeddings + Lexical matching + Source-Aware boosting retrieves relevant legal sections.',
        color: '#2e7d32',
    },
    {
        icon: <VerifiedIcon sx={{ fontSize: 36 }} />,
        title: 'Step 4: Rerank & Generate',
        desc: 'Cross-Encoder Reranker scores results. Transformer Decoder generates a fluent answer with cross-attention to context.',
        color: '#e65100',
    },
];

const models = [
    {
        name: 'Transformer Encoder',
        file: 'transformer_encoder.pth',
        arch: '4-layer Transformer with RoPE + GQA + SwiGLU + RMSNorm',
        purpose: 'Converts text into 256-dim context-aware embeddings for semantic search',
        training: 'Contrastive learning (InfoNCE) on corpus.txt — 5M params',
        color: '#1565c0',
    },
    {
        name: 'Cross-Encoder Reranker',
        file: 'reranker.pth',
        arch: 'Transformer Encoder + Classification Head',
        purpose: 'Jointly scores query-document pairs for precise relevance reranking (98.7% accuracy)',
        training: 'Binary classification on qa_data.json — 5M params',
        color: '#7b1fa2',
    },
    {
        name: 'Transformer Decoder',
        file: 'transformer_decoder.pth',
        arch: 'Causal Self-Attention + Cross-Attention + SwiGLU + Weight Tying',
        purpose: 'Generates fluent answers with attention to retrieved legal context (up to 200 tokens)',
        training: 'Next-token prediction on qa_data.json — 6M params',
        color: '#2e7d32',
    },
    {
        name: 'Intent Classifier',
        file: 'transformer_intent.pth',
        arch: '2-layer Transformer Encoder + Linear Head',
        purpose: 'Classifies query as legal question or greeting — 100% training accuracy',
        training: 'Supervised on intent_data.json — 3.5M params',
        color: '#e65100',
    },
];

const AboutPage = () => {
    return (
        <Box sx={{ flex: 1, overflow: 'auto', bgcolor: 'background.default' }}>

            {/* Header */}
            <Box sx={{ textAlign: 'center', py: 5, px: 3 }}>
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
                    <Box sx={{ display: 'flex', justifyContent: 'center', mb: 2 }}>
                        <AnimatedAvatar size={70} iconSize={40} />
                    </Box>
                    <Typography variant="h4" sx={{ fontWeight: 800, color: 'text.primary', mb: 1 }}>
                        How It Works
                    </Typography>
                    <Typography variant="body1" sx={{ color: 'text.secondary', maxWidth: 600, mx: 'auto' }}>
                        MineLawHub uses 4 custom-built Transformer neural networks (19.6M parameters total) — trained from scratch on mining law data. No external AI, no APIs.
                    </Typography>
                </motion.div>
            </Box>

            {/* 4-Step Flow */}
            <Box sx={{ maxWidth: 1000, mx: 'auto', px: 3, pb: 5 }}>
                <Grid container spacing={2}>
                    {steps.map((step, i) => (
                        <Grid item xs={12} sm={6} md={3} key={step.title}>
                            <motion.div
                                initial={{ opacity: 0, y: 30 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: i * 0.15 }}
                            >
                                <Paper
                                    elevation={0}
                                    sx={{
                                        p: 3,
                                        height: '100%',
                                        textAlign: 'center',
                                        border: '1px solid',
                                        borderColor: 'divider',
                                        borderRadius: 3,
                                        borderTop: `4px solid ${step.color}`,
                                    }}
                                >
                                    <Box sx={{ color: step.color, mb: 1 }}>{step.icon}</Box>
                                    <Typography variant="subtitle1" sx={{ fontWeight: 700, mb: 1, color: 'text.primary', fontSize: '0.95rem' }}>
                                        {step.title}
                                    </Typography>
                                    <Typography variant="body2" sx={{ color: 'text.secondary', lineHeight: 1.5 }}>
                                        {step.desc}
                                    </Typography>
                                </Paper>
                            </motion.div>
                        </Grid>
                    ))}
                </Grid>
            </Box>

            {/* Our 4 Custom Transformer Models */}
            <Box sx={{ bgcolor: 'background.paper', py: 5, px: 3, borderTop: '1px solid', borderBottom: '1px solid', borderColor: 'divider' }}>
                <Box sx={{ maxWidth: 1000, mx: 'auto' }}>
                    <Typography variant="h5" sx={{ fontWeight: 700, textAlign: 'center', mb: 1, color: 'text.primary' }}>
                        <MemoryIcon sx={{ mr: 1, verticalAlign: 'middle', color: 'secondary.main' }} />
                        Our 4 Custom Transformer Models (2025 Architecture)
                    </Typography>
                    <Typography variant="body2" sx={{ textAlign: 'center', color: 'text.secondary', mb: 4 }}>
                        All built from scratch using PyTorch with modern innovations: RoPE, GQA, SwiGLU, RMSNorm, BPE Tokenizer. ~19.6M parameters total. No pre-trained weights.
                    </Typography>

                    <Grid container spacing={3}>
                        {models.map((m, i) => (
                            <Grid item xs={12} md={6} key={m.name}>
                                <motion.div
                                    initial={{ opacity: 0, y: 30 }}
                                    animate={{ opacity: 1, y: 0 }}
                                    transition={{ delay: 0.4 + i * 0.15 }}
                                >
                                    <Paper
                                        elevation={0}
                                        sx={{
                                            p: 3,
                                            height: '100%',
                                            border: '1px solid',
                                            borderColor: 'divider',
                                            borderRadius: 3,
                                            borderLeft: `4px solid ${m.color}`,
                                        }}
                                    >
                                        <Typography variant="h6" sx={{ fontWeight: 700, color: m.color, mb: 1, fontSize: '1rem' }}>
                                            {m.name}
                                        </Typography>
                                        <Chip label={m.file} size="small" sx={{ mb: 1.5, fontFamily: 'monospace', fontSize: '0.75rem' }} />
                                        <Typography variant="body2" sx={{ color: 'text.secondary', mb: 0.5 }}>
                                            <strong>Architecture:</strong> {m.arch}
                                        </Typography>
                                        <Typography variant="body2" sx={{ color: 'text.secondary', mb: 0.5 }}>
                                            <strong>Purpose:</strong> {m.purpose}
                                        </Typography>
                                        <Typography variant="body2" sx={{ color: 'text.secondary' }}>
                                            <strong>Training:</strong> {m.training}
                                        </Typography>
                                    </Paper>
                                </motion.div>
                            </Grid>
                        ))}
                    </Grid>
                </Box>
            </Box>

            {/* Privacy Section */}
            <Box sx={{ maxWidth: 1000, mx: 'auto', py: 5, px: 3 }}>
                <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={{ delay: 0.6 }}>
                    <Paper
                        elevation={0}
                        sx={{
                            p: 4,
                            border: '1px solid',
                            borderColor: 'divider',
                            borderRadius: 3,
                            textAlign: 'center',
                            background: (theme) =>
                                theme.palette.mode === 'light'
                                    ? 'linear-gradient(135deg, #e8f5e9 0%, #fff3e0 100%)'
                                    : 'linear-gradient(135deg, #1b5e20 0%, #e65100 100%)',
                        }}
                    >
                        <SecurityIcon sx={{ fontSize: 50, color: 'success.main', mb: 1 }} />
                        <Typography variant="h5" sx={{ fontWeight: 700, mb: 1, color: 'text.primary' }}>
                            100% Offline & Private
                        </Typography>
                        <Typography variant="body1" sx={{ color: 'text.secondary', maxWidth: 600, mx: 'auto', mb: 2 }}>
                            No API keys. No internet. No cloud. Our custom Transformer models (~75MB total) run locally on your machine using PyTorch.
                            Your legal queries never leave your computer.
                        </Typography>
                        <Box sx={{ display: 'flex', justifyContent: 'center', gap: 1, flexWrap: 'wrap' }}>
                            <Chip label="🚫 No OpenAI" variant="outlined" />
                            <Chip label="🚫 No Gemini" variant="outlined" />
                            <Chip label="🚫 No Cloud AI" variant="outlined" />
                            <Chip label="✅ 100% Local" color="success" />
                        </Box>
                    </Paper>
                </motion.div>
            </Box>

            {/* Team Section */}
            <Box sx={{ textAlign: 'center', py: 4, px: 3, borderTop: '1px solid', borderColor: 'divider' }}>
                <Typography variant="h6" sx={{ fontWeight: 700, color: 'text.primary', mb: 2 }}>
                    Our Team
                </Typography>
                <Box sx={{ display: 'flex', justifyContent: 'center', gap: 4, flexWrap: 'wrap' }}>
                    <Paper elevation={0} sx={{ p: 3, border: '1px solid', borderColor: 'divider', borderRadius: 3, minWidth: 220 }}>
                        <Typography variant="subtitle1" sx={{ fontWeight: 700, color: 'text.primary' }}>
                            Madhankumar S
                        </Typography>
                        <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                            7376222IT184
                        </Typography>
                        <Typography variant="body2" sx={{ color: 'text.secondary', mt: 1 }}>
                            Backend, Custom Neural Networks, Search Engine, System Architecture
                        </Typography>
                    </Paper>
                    <Paper elevation={0} sx={{ p: 3, border: '1px solid', borderColor: 'divider', borderRadius: 3, minWidth: 220 }}>
                        <Typography variant="subtitle1" sx={{ fontWeight: 700, color: 'text.primary' }}>
                            Jayatchana Aravind M
                        </Typography>
                        <Typography variant="caption" sx={{ color: 'text.secondary' }}>
                            7376222IT159
                        </Typography>
                        <Typography variant="body2" sx={{ color: 'text.secondary', mt: 1 }}>
                            React Frontend, Data Collection, Law Validation, Testing & Verification
                        </Typography>
                    </Paper>
                </Box>
            </Box>

            {/* Footer */}
            <Box sx={{ textAlign: 'center', py: 3, borderTop: '1px solid', borderColor: 'divider' }}>
                <Typography variant="body2" color="text.secondary">
                    MineLawHub — Custom Transformer Architecture (2025) | 19.6M Parameters | 100% Proprietary Models
                </Typography>
            </Box>
        </Box>
    );
};

export default AboutPage;
