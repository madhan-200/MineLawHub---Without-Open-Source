import React, { useState, useEffect, useCallback } from 'react';
import { Box, Snackbar, Alert } from '@mui/material';
import ChatWindow from '../components/ChatWindow';
import InputBox from '../components/InputBox';
import { useSearchParams } from 'react-router-dom';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

// localStorage helpers
const HISTORY_KEY = 'minelawhub_history';
const ACTIVE_KEY = 'minelawhub_active';

const generateId = () => Date.now().toString(36) + Math.random().toString(36).slice(2, 7);

const loadHistory = () => {
    try {
        return JSON.parse(localStorage.getItem(HISTORY_KEY) || '[]');
    } catch { return []; }
};

const saveHistory = (history) => {
    localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
};

const ChatPage = () => {
    const [messages, setMessages] = useState([]);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [sessionId, setSessionId] = useState(() => {
        return localStorage.getItem(ACTIVE_KEY) || generateId();
    });
    const [searchParams, setSearchParams] = useSearchParams();

    // Load active session on mount
    useEffect(() => {
        const history = loadHistory();
        const active = history.find(h => h.id === sessionId);
        if (active && active.messages.length > 0) {
            setMessages(active.messages);
        }
        localStorage.setItem(ACTIVE_KEY, sessionId);
    }, [sessionId]);

    // Handle prefilled query from Laws page
    useEffect(() => {
        const prefill = searchParams.get('q');
        if (prefill) {
            setSearchParams({}, { replace: true });
            // Small delay to let component mount
            setTimeout(() => handleSendMessage(prefill), 300);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    // Save messages to localStorage whenever they change
    useEffect(() => {
        if (messages.length === 0) return;
        const history = loadHistory();
        const idx = history.findIndex(h => h.id === sessionId);
        const title = messages.find(m => m.isUser)?.text?.slice(0, 60) || 'New Chat';
        const entry = {
            id: sessionId,
            title,
            timestamp: Date.now(),
            messages,
        };
        if (idx >= 0) {
            history[idx] = entry;
        } else {
            history.unshift(entry);
        }
        // Keep only last 20 conversations
        saveHistory(history.slice(0, 20));
    }, [messages, sessionId]);

    const handleSendMessage = useCallback(async (text) => {
        const userMessage = { text, isUser: true };
        setMessages(prev => [...prev, userMessage]);
        setIsLoading(true);
        setError(null);

        try {
            const response = await fetch(`${API_BASE_URL}/chat`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ query: text }),
            });

            if (!response.ok) throw new Error(`API error: ${response.status}`);

            const data = await response.json();
            const aiMessage = {
                text: data.answer,
                isUser: false,
                citations: data.citations,
                intent: data.intent,
            };
            setMessages(prev => [...prev, aiMessage]);
        } catch (err) {
            console.error('Error sending message:', err);
            setError(err.message || 'Failed to get response.');
            const errorMessage = {
                text: 'Sorry, I encountered an error. Please make sure the backend server is running.',
                isUser: false,
            };
            setMessages(prev => [...prev, errorMessage]);
        } finally {
            setIsLoading(false);
        }
    }, []);

    const handleNewChat = useCallback(() => {
        const newId = generateId();
        setSessionId(newId);
        setMessages([]);
        localStorage.setItem(ACTIVE_KEY, newId);
    }, []);

    const handleOpenSession = useCallback((id) => {
        const history = loadHistory();
        const session = history.find(h => h.id === id);
        if (session) {
            setSessionId(id);
            setMessages(session.messages);
            localStorage.setItem(ACTIVE_KEY, id);
        }
    }, []);

    const handleClearHistory = useCallback(() => {
        saveHistory([]);
        handleNewChat();
    }, [handleNewChat]);

    return (
        <Box sx={{ display: 'flex', flexDirection: 'column', flex: 1, height: '100%', overflow: 'hidden' }}>
            <ChatWindow messages={messages} isLoading={isLoading} />
            <InputBox onSendMessage={handleSendMessage} isLoading={isLoading} />
            <Snackbar
                open={!!error}
                autoHideDuration={6000}
                onClose={() => setError(null)}
                anchorOrigin={{ vertical: 'top', horizontal: 'center' }}
            >
                <Alert onClose={() => setError(null)} severity="error" sx={{ width: '100%' }}>
                    {error}
                </Alert>
            </Snackbar>
        </Box>
    );
};

// Export helpers for Sidebar to use
export { loadHistory, saveHistory, generateId };
export default ChatPage;
