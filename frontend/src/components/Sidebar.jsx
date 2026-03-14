import React, { useState, useEffect } from 'react';
import {
    Drawer, List, ListItem, ListItemIcon, ListItemText, Typography, Box, Divider,
    IconButton, Button, ListItemButton
} from '@mui/material';
import { useNavigate, useLocation } from 'react-router-dom';
import HomeIcon from '@mui/icons-material/Home';
import ChatIcon from '@mui/icons-material/Chat';
import GavelIcon from '@mui/icons-material/Gavel';
import InfoIcon from '@mui/icons-material/Info';
import AddIcon from '@mui/icons-material/Add';
import HistoryIcon from '@mui/icons-material/History';
import DeleteSweepIcon from '@mui/icons-material/DeleteSweep';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import { loadHistory, saveHistory } from '../pages/ChatPage';

const drawerWidth = 280;

const navItems = [
    { label: 'Home', icon: <HomeIcon />, path: '/' },
    { label: 'Chat', icon: <ChatIcon />, path: '/chat' },
    { label: 'Laws Reference', icon: <GavelIcon />, path: '/laws' },
    { label: 'How It Works', icon: <InfoIcon />, path: '/about' },
];

const quickAsk = [
    'What are penalties for violating mining laws?',
    'Who can work in mines?',
    'What are safety rules in mines?',
    'What is the Mines Act 1952?',
];

const Sidebar = ({ open, onClose }) => {
    const navigate = useNavigate();
    const location = useLocation();
    const [history, setHistory] = useState([]);

    useEffect(() => {
        if (open) {
            setHistory(loadHistory());
        }
    }, [open]);

    const handleNav = (path) => {
        navigate(path);
        onClose();
    };

    const handleQuickAsk = (query) => {
        navigate(`/chat?q=${encodeURIComponent(query)}`);
        onClose();
    };

    const handleOpenSession = (id) => {
        // Set the active session and navigate to chat
        localStorage.setItem('minelawhub_active', id);
        navigate('/chat');
        onClose();
        // Force reload to pick up the session
        window.location.reload();
    };

    const handleClearHistory = () => {
        saveHistory([]);
        setHistory([]);
    };

    const handleNewChat = () => {
        const newId = Date.now().toString(36) + Math.random().toString(36).slice(2, 7);
        localStorage.setItem('minelawhub_active', newId);
        navigate('/chat');
        onClose();
        window.location.reload();
    };

    const timeAgo = (ts) => {
        const diff = Date.now() - ts;
        const mins = Math.floor(diff / 60000);
        if (mins < 1) return 'Just now';
        if (mins < 60) return `${mins}m ago`;
        const hrs = Math.floor(mins / 60);
        if (hrs < 24) return `${hrs}h ago`;
        const days = Math.floor(hrs / 24);
        return `${days}d ago`;
    };

    return (
        <Drawer
            variant="temporary"
            anchor="left"
            open={open}
            onClose={onClose}
            sx={{
                '& .MuiDrawer-paper': {
                    width: drawerWidth,
                    boxSizing: 'border-box',
                    bgcolor: 'background.paper',
                    borderRight: '1px solid',
                    borderColor: 'divider',
                },
            }}
        >
            {/* Header */}
            <Box sx={{ display: 'flex', alignItems: 'center', p: 2, justifyContent: 'space-between' }}>
                <Typography variant="h6" color="primary" sx={{ fontWeight: 700 }}>
                    MineLawHub
                </Typography>
                <IconButton onClick={onClose}>
                    <ChevronLeftIcon />
                </IconButton>
            </Box>

            {/* New Chat Button */}
            <Box sx={{ px: 2, pb: 1 }}>
                <Button
                    fullWidth
                    variant="outlined"
                    startIcon={<AddIcon />}
                    onClick={handleNewChat}
                    sx={{
                        borderColor: 'secondary.main',
                        color: 'secondary.main',
                        fontWeight: 600,
                        borderRadius: 2,
                        '&:hover': { bgcolor: 'secondary.main', color: 'primary.dark' },
                    }}
                >
                    New Chat
                </Button>
            </Box>

            <Divider />

            {/* Navigation */}
            <List dense subheader={<Box sx={{ px: 2, pt: 1.5, pb: 0.5, fontWeight: 600, fontSize: '0.75rem', color: 'text.secondary', textTransform: 'uppercase' }}>Navigation</Box>}>
                {navItems.map((item) => (
                    <ListItemButton
                        key={item.path}
                        selected={location.pathname === item.path}
                        onClick={() => handleNav(item.path)}
                        sx={{
                            mx: 1,
                            borderRadius: 2,
                            mb: 0.5,
                            '&.Mui-selected': { bgcolor: 'primary.main', color: 'white', '& .MuiListItemIcon-root': { color: 'white' } },
                            '&.Mui-selected:hover': { bgcolor: 'primary.dark' },
                        }}
                    >
                        <ListItemIcon sx={{ minWidth: 36 }}>{item.icon}</ListItemIcon>
                        <ListItemText primary={item.label} primaryTypographyProps={{ fontSize: '0.9rem', fontWeight: 500 }} />
                    </ListItemButton>
                ))}
            </List>

            <Divider />

            {/* Quick Ask */}
            <List dense subheader={<Box sx={{ px: 2, pt: 1.5, pb: 0.5, fontWeight: 600, fontSize: '0.75rem', color: 'text.secondary', textTransform: 'uppercase' }}>Quick Ask</Box>}>
                {quickAsk.map((q) => (
                    <ListItemButton
                        key={q}
                        onClick={() => handleQuickAsk(q)}
                        sx={{ mx: 1, borderRadius: 2, mb: 0.3 }}
                    >
                        <ListItemIcon sx={{ minWidth: 32 }}>
                            <AutoAwesomeIcon fontSize="small" color="secondary" />
                        </ListItemIcon>
                        <ListItemText
                            primary={q.length > 35 ? q.slice(0, 35) + '...' : q}
                            primaryTypographyProps={{ fontSize: '0.8rem' }}
                        />
                    </ListItemButton>
                ))}
            </List>

            <Divider />

            {/* History */}
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', px: 2, pt: 1.5, pb: 0.5 }}>
                <Typography sx={{ fontWeight: 600, fontSize: '0.75rem', color: 'text.secondary', textTransform: 'uppercase' }}>
                    History
                </Typography>
                {history.length > 0 && (
                    <IconButton size="small" onClick={handleClearHistory} title="Clear all history">
                        <DeleteSweepIcon fontSize="small" />
                    </IconButton>
                )}
            </Box>
            <List dense sx={{ flex: 1, overflow: 'auto' }}>
                {history.length === 0 ? (
                    <ListItem>
                        <ListItemText
                            primary="No conversations yet"
                            primaryTypographyProps={{ fontSize: '0.8rem', color: 'text.secondary', fontStyle: 'italic' }}
                        />
                    </ListItem>
                ) : (
                    history.slice(0, 10).map((h) => (
                        <ListItemButton
                            key={h.id}
                            onClick={() => handleOpenSession(h.id)}
                            sx={{ mx: 1, borderRadius: 2, mb: 0.3 }}
                        >
                            <ListItemIcon sx={{ minWidth: 32 }}>
                                <HistoryIcon fontSize="small" />
                            </ListItemIcon>
                            <ListItemText
                                primary={h.title?.length > 30 ? h.title.slice(0, 30) + '...' : h.title}
                                secondary={timeAgo(h.timestamp)}
                                primaryTypographyProps={{ fontSize: '0.8rem', fontWeight: 500 }}
                                secondaryTypographyProps={{ fontSize: '0.7rem' }}
                            />
                        </ListItemButton>
                    ))
                )}
            </List>
        </Drawer>
    );
};

export default Sidebar;
