import React, { useRef, useEffect } from 'react';
import { Box, Paper, Typography } from '@mui/material';
import MessageBubble from './MessageBubble';
import AnimatedAvatar from './AnimatedAvatar';
import TypewriterText from './TypewriterText';
import { motion } from 'framer-motion';

const ChatWindow = ({ messages, isLoading }) => {
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    };

    useEffect(() => {
        scrollToBottom();
    }, [messages]);

    return (
        <Paper
            elevation={0}
            sx={{
                flex: 1,
                overflow: 'auto',
                p: 3,
                bgcolor: 'background.default',
                display: 'flex',
                flexDirection: 'column',
            }}
        >
            {messages.length === 0 ? (
                <Box
                    sx={{
                        flex: 1,
                        display: 'flex',
                        flexDirection: 'column',
                        alignItems: 'center',
                        justifyContent: 'center',
                        textAlign: 'center',
                        gap: 2,
                    }}
                >
                    <AnimatedAvatar size={80} iconSize={48} />

                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.8, ease: "easeOut" }}
                    >
                        <Typography variant="h4" color="text.primary" gutterBottom sx={{ fontWeight: 800 }}>
                            Welcome to MineLawHub
                        </Typography>
                    </motion.div>

                    <TypewriterText
                        text="Your AI-powered assistant for Indian Mining Laws, Acts, Rules, and Regulations. Ask me anything about mining legislation, recent updates, or specific regulations."
                        color="text.primary"
                        sx={{ maxWidth: 650, fontSize: '1.1rem' }}
                    />

                    <Box sx={{ mt: 2, display: 'flex', flexWrap: 'wrap', gap: 1, justifyContent: 'center' }}>
                        <Typography variant="caption" color="text.primary" sx={{ fontWeight: 600 }}>
                            Try asking:
                        </Typography>
                        <Typography variant="caption" sx={{ fontStyle: 'italic', color: 'secondary.main' }}>
                            "What are the safety regulations for underground mining?"
                        </Typography>
                        <Typography variant="caption" sx={{ fontStyle: 'italic', color: 'secondary.main' }}>
                            "Recent DGMS circulars on mine safety"
                        </Typography>
                    </Box>
                </Box>
            ) : (
                <>
                    {messages.map((msg, index) => (
                        <MessageBubble
                            key={index}
                            message={msg}
                            isUser={msg.isUser}
                        />
                    ))}
                    {isLoading && (
                        <Box sx={{ display: 'flex', p: 2 }}>
                            <Box sx={{ mr: 2 }}>
                                <AnimatedAvatar size={40} iconSize={24} isThinking={true} />
                            </Box>
                            <Paper
                                elevation={1}
                                sx={{
                                    p: 2,
                                    bgcolor: 'background.paper',
                                    borderRadius: 2,
                                    borderBottomLeftRadius: 0,
                                }}
                            >
                                <Box sx={{ display: 'flex', gap: 0.5, alignItems: 'center', height: 24 }}>
                                    {[0, 1, 2].map((i) => (
                                        <Box
                                            key={i}
                                            sx={{
                                                width: 8,
                                                height: 8,
                                                bgcolor: 'primary.light',
                                                borderRadius: '50%',
                                                animation: 'bounce 1.4s infinite ease-in-out both',
                                                animationDelay: `${i * 0.16}s`,
                                                '@keyframes bounce': {
                                                    '0%, 80%, 100%': { transform: 'scale(0)' },
                                                    '40%': { transform: 'scale(1)' },
                                                },
                                            }}
                                        />
                                    ))}
                                </Box>
                            </Paper>
                        </Box>
                    )}
                    <div ref={messagesEndRef} />
                </>
            )}
        </Paper>
    );
};

export default ChatWindow;
