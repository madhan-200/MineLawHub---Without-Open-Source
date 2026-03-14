import React, { useState } from 'react';
import { Box, TextField, IconButton, CircularProgress, Chip, Fade } from '@mui/material';
import SendIcon from '@mui/icons-material/Send';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';

const SUGGESTIONS = [
    "Safety regulations for coal mines",
    "Recent DGMS circulars",
    "Duties of Safety Officer",
    "Winding engine rules"
];

const InputBox = ({ onSendMessage, isLoading }) => {
    const [message, setMessage] = useState('');

    const handleSubmit = (e) => {
        e.preventDefault();

        if (message.trim() && !isLoading) {
            onSendMessage(message.trim());
            setMessage('');
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSubmit(e);
        }
    };

    return (
        <Box
            component="form"
            onSubmit={handleSubmit}
            sx={{
                display: 'flex',
                flexDirection: 'column',
                gap: 2,
                p: 2,
                bgcolor: 'background.paper',
                borderTop: '1px solid rgba(0,0,0,0.1)',
            }}
        >
            <Fade in={!isLoading && !message}>
                <Box sx={{ display: 'flex', gap: 1, overflowX: 'auto', pb: 1, '&::-webkit-scrollbar': { display: 'none' } }}>
                    {SUGGESTIONS.map((s) => (
                        <Chip
                            key={s}
                            icon={<AutoAwesomeIcon fontSize="small" />}
                            label={s}
                            onClick={() => onSendMessage(s)}
                            clickable
                            color="secondary"
                            variant="outlined"
                            sx={{ borderRadius: 4, bgcolor: 'background.default' }}
                        />
                    ))}
                </Box>
            </Fade>

            <Box sx={{ display: 'flex', gap: 1 }}>
                <TextField
                    fullWidth
                    multiline
                    maxRows={4}
                    value={message}
                    onChange={(e) => setMessage(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Ask about mining laws, regulations, or recent updates..."
                    disabled={isLoading}
                    variant="outlined"
                    sx={{
                        '& .MuiOutlinedInput-root': {
                            borderRadius: 3,
                        },
                    }}
                />
                <IconButton
                    type="submit"
                    color="primary"
                    disabled={!message.trim() || isLoading}
                    sx={{
                        bgcolor: 'primary.main',
                        color: 'white',
                        '&:hover': {
                            bgcolor: 'primary.dark',
                        },
                        '&:disabled': {
                            bgcolor: 'action.disabledBackground',
                        },
                        width: 56,
                        height: 56,
                    }}
                >
                    {isLoading ? <CircularProgress size={24} color="inherit" /> : <SendIcon />}
                </IconButton>
            </Box>
        </Box>
    );
};

export default InputBox;
