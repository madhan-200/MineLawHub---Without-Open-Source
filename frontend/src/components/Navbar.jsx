import React, { useContext } from 'react';
import { AppBar, Toolbar, Typography, Chip, IconButton, useTheme, Button, Box } from '@mui/material';
import GavelIcon from '@mui/icons-material/Gavel';
import Brightness4Icon from '@mui/icons-material/Brightness4';
import Brightness7Icon from '@mui/icons-material/Brightness7';
import MenuIcon from '@mui/icons-material/Menu';
import { useNavigate, useLocation } from 'react-router-dom';
import { ColorModeContext } from '../App';

const navLinks = [
    { label: 'Home', path: '/' },
    { label: 'Chat', path: '/chat' },
    { label: 'Laws', path: '/laws' },
    { label: 'About', path: '/about' },
];

const Navbar = ({ onMenuClick }) => {
    const theme = useTheme();
    const colorMode = useContext(ColorModeContext);
    const navigate = useNavigate();
    const location = useLocation();

    return (
        <AppBar position="static" elevation={2}>
            <Toolbar>
                <IconButton
                    size="large"
                    edge="start"
                    color="inherit"
                    aria-label="menu"
                    sx={{ mr: 1 }}
                    onClick={onMenuClick}
                >
                    <MenuIcon />
                </IconButton>
                <GavelIcon sx={{ mr: 1.5, fontSize: 28 }} />
                <Typography
                    variant="h6"
                    component="div"
                    sx={{ fontWeight: 700, cursor: 'pointer', mr: 3 }}
                    onClick={() => navigate('/')}
                >
                    MineLawHub
                </Typography>

                {/* Nav Links — hidden on xs */}
                <Box sx={{ display: { xs: 'none', md: 'flex' }, gap: 0.5, flexGrow: 1 }}>
                    {navLinks.map((link) => (
                        <Button
                            key={link.path}
                            size="small"
                            onClick={() => navigate(link.path)}
                            sx={{
                                color: location.pathname === link.path ? 'secondary.main' : 'inherit',
                                fontWeight: location.pathname === link.path ? 700 : 500,
                                textTransform: 'none',
                                fontSize: '0.9rem',
                                borderBottom: location.pathname === link.path ? '2px solid' : '2px solid transparent',
                                borderColor: location.pathname === link.path ? 'secondary.main' : 'transparent',
                                borderRadius: 0,
                                px: 1.5,
                            }}
                        >
                            {link.label}
                        </Button>
                    ))}
                </Box>

                <Box sx={{ display: { xs: 'flex', md: 'none' }, flexGrow: 1 }} />

                <Chip
                    label="AI-Powered"
                    color="secondary"
                    size="small"
                    sx={{ fontWeight: 500, mr: 1 }}
                />
                <IconButton sx={{ ml: 0.5 }} onClick={colorMode.toggleColorMode} color="inherit">
                    {theme.palette.mode === 'dark' ? <Brightness7Icon /> : <Brightness4Icon />}
                </IconButton>
            </Toolbar>
        </AppBar>
    );
};

export default Navbar;
