import React from 'react';
import { Box, Typography, Button, Grid, Paper, Chip } from '@mui/material';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import SmartToyIcon from '@mui/icons-material/SmartToy';
import SearchIcon from '@mui/icons-material/Search';
import SecurityIcon from '@mui/icons-material/Security';
import FormatQuoteIcon from '@mui/icons-material/FormatQuote';
import AnimatedAvatar from '../components/AnimatedAvatar';

const stats = [
    { label: '6,281 Legal Chunks Indexed', color: '#ffc107' },
    { label: '4 Custom Transformer Models', color: '#4caf50' },
    { label: '6 Mining Laws Covered', color: '#2196f3' },
    { label: '100% Offline & Private', color: '#f44336' },
];

const features = [
    {
        icon: <SmartToyIcon sx={{ fontSize: 40 }} />,
        title: 'Custom Transformer AI',
        desc: 'Four Transformer neural networks (19.6M params) built from scratch using PyTorch — BPE Tokenizer, Transformer Encoder, Cross-Encoder Reranker, and Transformer Decoder. No external APIs.',
    },
    {
        icon: <SearchIcon sx={{ fontSize: 40 }} />,
        title: 'Hybrid Search + Reranking',
        desc: '256-dim Transformer embeddings + lexical matching + source-aware boosting, followed by Cross-Encoder reranking (98.7% accuracy) for precise results.',
    },
    {
        icon: <SecurityIcon sx={{ fontSize: 40 }} />,
        title: 'Privacy First',
        desc: 'Runs entirely on your machine. Zero data sent to any server. No API keys, no internet required. Your queries stay private.',
    },
    {
        icon: <FormatQuoteIcon sx={{ fontSize: 40 }} />,
        title: 'Source Citations',
        desc: 'Every answer includes the Act name and Section number. Supporting legal sections displayed for verification.',
    },
];

const HomePage = () => {
    const navigate = useNavigate();

    return (
        <Box sx={{ flex: 1, overflow: 'auto', bgcolor: 'background.default' }}>
            {/* Hero Section */}
            <Box
                sx={{
                    textAlign: 'center',
                    py: { xs: 6, md: 10 },
                    px: 3,
                    background: (theme) =>
                        theme.palette.mode === 'light'
                            ? 'linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%)'
                            : 'linear-gradient(135deg, #0f172a 0%, #1e293b 100%)',
                    color: 'white',
                }}
            >
                <motion.div
                    initial={{ opacity: 0, y: 30 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.8 }}
                >
                    <Box sx={{ display: 'flex', justifyContent: 'center', mb: 3 }}>
                        <AnimatedAvatar size={90} iconSize={50} />
                    </Box>
                    <Typography
                        variant="h3"
                        sx={{
                            fontWeight: 800,
                            mb: 2,
                            background: 'linear-gradient(90deg, #ffc107, #ffca28, #ffd54f)',
                            backgroundClip: 'text',
                            WebkitBackgroundClip: 'text',
                            WebkitTextFillColor: 'transparent',
                        }}
                    >
                        MineLawHub
                    </Typography>
                    <Typography variant="h6" sx={{ mb: 1, fontWeight: 400, opacity: 0.9 }}>
                        AI-Powered Chatbot for Indian Mining Laws
                    </Typography>
                    <Typography variant="body1" sx={{ mb: 4, opacity: 0.7, maxWidth: 600, mx: 'auto' }}>
                        Get instant, accurate answers about the Mines Act 1952, MCDR 2017, Coal Mines Regulations,
                        and more — powered by our custom Transformer architecture (2025) with 19.6M parameters.
                    </Typography>

                    <Button
                        variant="contained"
                        size="large"
                        onClick={() => navigate('/chat')}
                        sx={{
                            bgcolor: '#ffc107',
                            color: '#0f172a',
                            fontWeight: 700,
                            px: 5,
                            py: 1.5,
                            fontSize: '1.1rem',
                            borderRadius: 3,
                            '&:hover': { bgcolor: '#ffca28', transform: 'scale(1.05)' },
                            transition: 'all 0.3s ease',
                        }}
                    >
                        Start Chatting →
                    </Button>
                </motion.div>
            </Box>

            {/* Stats Strip */}
            <Box
                sx={{
                    display: 'flex',
                    justifyContent: 'center',
                    flexWrap: 'wrap',
                    gap: 2,
                    py: 3,
                    px: 2,
                    bgcolor: 'background.paper',
                    borderBottom: '1px solid',
                    borderColor: 'divider',
                }}
            >
                {stats.map((stat, i) => (
                    <motion.div
                        key={stat.label}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: 0.3 + i * 0.15 }}
                    >
                        <Chip
                            label={stat.label}
                            sx={{
                                fontWeight: 600,
                                fontSize: '0.9rem',
                                py: 2.5,
                                px: 1,
                                bgcolor: stat.color + '18',
                                color: stat.color,
                                border: `1px solid ${stat.color}40`,
                            }}
                        />
                    </motion.div>
                ))}
            </Box>

            {/* Feature Cards */}
            <Box sx={{ maxWidth: 1000, mx: 'auto', py: 6, px: 3 }}>
                <Typography
                    variant="h5"
                    sx={{ fontWeight: 700, textAlign: 'center', mb: 4, color: 'text.primary' }}
                >
                    What Makes MineLawHub Different
                </Typography>
                <Grid container spacing={3}>
                    {features.map((f, i) => (
                        <Grid item xs={12} sm={6} key={f.title}>
                            <motion.div
                                initial={{ opacity: 0, y: 30 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: 0.5 + i * 0.15 }}
                            >
                                <Paper
                                    elevation={0}
                                    sx={{
                                        p: 3,
                                        height: '100%',
                                        border: '1px solid',
                                        borderColor: 'divider',
                                        borderRadius: 3,
                                        transition: 'all 0.3s ease',
                                        '&:hover': {
                                            borderColor: 'secondary.main',
                                            transform: 'translateY(-4px)',
                                            boxShadow: '0 8px 30px rgba(0,0,0,0.12)',
                                        },
                                    }}
                                >
                                    <Box sx={{ color: 'secondary.main', mb: 2 }}>{f.icon}</Box>
                                    <Typography variant="h6" sx={{ fontWeight: 700, mb: 1, color: 'text.primary' }}>
                                        {f.title}
                                    </Typography>
                                    <Typography variant="body2" sx={{ color: 'text.secondary', lineHeight: 1.6 }}>
                                        {f.desc}
                                    </Typography>
                                </Paper>
                            </motion.div>
                        </Grid>
                    ))}
                </Grid>
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

export default HomePage;
