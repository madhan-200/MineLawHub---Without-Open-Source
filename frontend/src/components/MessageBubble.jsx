import React from 'react';
import { Box, Paper, Typography, Chip, Link, IconButton, Tooltip, alpha } from '@mui/material';
import { motion } from 'framer-motion';
import PersonIcon from '@mui/icons-material/Person';
import OpenInNewIcon from '@mui/icons-material/OpenInNew';
import ContentCopyIcon from '@mui/icons-material/ContentCopy';
import ThumbUpAltIcon from '@mui/icons-material/ThumbUpAlt';
import ThumbDownAltIcon from '@mui/icons-material/ThumbDownAlt';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import AnimatedAvatar from './AnimatedAvatar';

const MessageBubble = ({ message, isUser }) => {
    const handleCopy = () => {
        navigator.clipboard.writeText(message.text);
    };

    const renderCitations = (citations) => {
        if (!citations || citations.length === 0) return null;

        return (
            <Box sx={{ mt: 2, pt: 2, borderTop: '1px solid rgba(0,0,0,0.1)' }}>
                <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary', mb: 1, display: 'block' }}>
                    Sources:
                </Typography>
                <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 1 }}>
                    {citations.map((citation, index) => (
                        <Chip
                            key={index}
                            label={citation.type === 'web' ? citation.title : `${citation.source} - ${citation.section}`}
                            size="small"
                            variant="outlined"
                            component={citation.url ? Link : 'div'}
                            href={citation.url || undefined}
                            target="_blank"
                            rel="noopener noreferrer"
                            clickable={!!citation.url}
                            icon={citation.url ? <OpenInNewIcon /> : undefined}
                            sx={{
                                maxWidth: '100%',
                                '& .MuiChip-label': { overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }
                            }}
                        />
                    ))}
                </Box>
            </Box>
        );
    };

    return (
        <Box
            component={motion.div}
            initial={{ opacity: 0, x: isUser ? 50 : -50, scale: 0.9 }}
            animate={{ opacity: 1, x: 0, scale: 1 }}
            transition={{ duration: 0.3, type: 'spring', stiffness: 100 }}
            sx={{
                display: 'flex',
                justifyContent: isUser ? 'flex-end' : 'flex-start',
                mb: 2,
            }}
        >
            <Box
                sx={{
                    display: 'flex',
                    flexDirection: isUser ? 'row-reverse' : 'row',
                    alignItems: 'flex-start',
                    maxWidth: '85%', // Slight increase for better reading
                    gap: 1.5,
                }}
            >
                <Box sx={{ flexShrink: 0 }}>
                    {isUser ? (
                        <Box
                            sx={{
                                width: 40,
                                height: 40,
                                borderRadius: '50%',
                                bgcolor: 'primary.main',
                                display: 'flex',
                                alignItems: 'center',
                                justifyContent: 'center',
                            }}
                        >
                            <PersonIcon sx={{ color: 'white' }} />
                        </Box>
                    ) : (
                        <AnimatedAvatar size={40} iconSize={24} />
                    )}
                </Box>

                <Paper
                    component={motion.div}
                    whileHover={{ scale: 1.01 }}
                    elevation={1}
                    sx={{
                        p: 2,
                        bgcolor: isUser ? 'primary.light' : 'background.paper',
                        color: isUser ? 'white' : 'text.primary',
                        borderRadius: 2,
                        borderTopLeftRadius: !isUser ? 0 : 2,
                        borderTopRightRadius: isUser ? 0 : 2,
                    }}
                >
                    <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                            p: ({ node, ...props }) => <Typography variant="body1" sx={{ mb: 1.5, '&:last-child': { mb: 0 } }} {...props} />,
                            h1: ({ node, ...props }) => <Typography variant="h5" sx={{ fontWeight: 700, mt: 2, mb: 1, color: isUser ? 'inherit' : 'primary.main' }} {...props} />,
                            h2: ({ node, ...props }) => <Typography variant="h6" sx={{ fontWeight: 600, mt: 2, mb: 1, color: isUser ? 'inherit' : 'primary.dark' }} {...props} />,
                            h3: ({ node, ...props }) => <Typography variant="subtitle1" sx={{ fontWeight: 600, mt: 1.5, mb: 0.5 }} {...props} />,
                            ul: ({ node, ...props }) => <Box component="ul" sx={{ pl: 2, mb: 1.5 }} {...props} />,
                            ol: ({ node, ...props }) => <Box component="ol" sx={{ pl: 2, mb: 1.5 }} {...props} />,
                            li: ({ node, ...props }) => <Typography component="li" variant="body1" sx={{ mb: 0.5 }} {...props} />,
                            strong: ({ node, ...props }) => <Box component="span" sx={{ fontWeight: 700 }} {...props} />,
                            table: ({ node, ...props }) => (
                                <Box sx={{ overflowX: 'auto', mb: 2, border: '1px solid', borderColor: isUser ? 'rgba(255,255,255,0.2)' : 'divider', borderRadius: 1 }}>
                                    <Box component="table" sx={{ width: '100%', borderCollapse: 'collapse', '& td, & th': { p: 1, borderBottom: '1px solid', borderColor: isUser ? 'rgba(255,255,255,0.2)' : 'divider', textAlign: 'left' }, '& th': { bgcolor: isUser ? 'rgba(255,255,255,0.1)' : 'action.hover', fontWeight: 600 } }} {...props} />
                                </Box>
                            ),
                            a: ({ node, ...props }) => <Link target="_blank" rel="noopener noreferrer" sx={{ color: isUser ? 'inherit' : 'secondary.main' }} {...props} />,
                            code: ({ node, inline, className, children, ...props }) => (
                                <Box component="code" sx={{ bgcolor: isUser ? 'rgba(255,255,255,0.1)' : 'action.hover', p: 0.5, borderRadius: 0.5, fontFamily: 'monospace', fontSize: '0.875rem' }} {...props}>
                                    {children}
                                </Box>
                            )
                        }}
                    >
                        {message.text}
                    </ReactMarkdown>

                    {!isUser && message.citations && renderCitations(message.citations)}

                    {!isUser && (
                        <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 1, gap: 0.5 }}>
                            <Tooltip title="Copy to clipboard">
                                <IconButton size="small" onClick={handleCopy}>
                                    <ContentCopyIcon fontSize="small" />
                                </IconButton>
                            </Tooltip>
                            <Tooltip title="Good response">
                                <IconButton size="small">
                                    <ThumbUpAltIcon fontSize="small" />
                                </IconButton>
                            </Tooltip>
                            <Tooltip title="Bad response">
                                <IconButton size="small">
                                    <ThumbDownAltIcon fontSize="small" />
                                </IconButton>
                            </Tooltip>
                        </Box>
                    )}
                </Paper>
            </Box>
        </Box>
    );
};

export default MessageBubble;
