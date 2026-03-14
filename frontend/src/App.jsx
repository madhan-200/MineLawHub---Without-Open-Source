import React, { useState, useMemo, createContext } from 'react';
import { ThemeProvider, createTheme } from '@mui/material/styles';
import { CssBaseline, Box } from '@mui/material';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import HomePage from './pages/HomePage';
import ChatPage from './pages/ChatPage';
import LawsPage from './pages/LawsPage';
import AboutPage from './pages/AboutPage';
import getDesignTokens from './theme/theme';

export const ColorModeContext = createContext({ toggleColorMode: () => { } });

function App() {
    const [mode, setMode] = useState('light');
    const [sidebarOpen, setSidebarOpen] = useState(false);

    const colorMode = useMemo(
        () => ({
            toggleColorMode: () => {
                setMode((prevMode) => (prevMode === 'light' ? 'dark' : 'light'));
            },
        }),
        [],
    );

    const theme = useMemo(() => createTheme(getDesignTokens(mode)), [mode]);

    return (
        <ColorModeContext.Provider value={colorMode}>
            <ThemeProvider theme={theme}>
                <CssBaseline />
                <BrowserRouter>
                    <Box
                        sx={{
                            display: 'flex',
                            height: '100vh',
                            bgcolor: 'background.default',
                        }}
                    >
                        <Sidebar
                            open={sidebarOpen}
                            onClose={() => setSidebarOpen(false)}
                        />

                        <Box sx={{ display: 'flex', flexDirection: 'column', flexGrow: 1, height: '100%', overflow: 'hidden' }}>
                            <Navbar onMenuClick={() => setSidebarOpen(!sidebarOpen)} />
                            <Routes>
                                <Route path="/" element={<HomePage />} />
                                <Route path="/chat" element={<ChatPage />} />
                                <Route path="/laws" element={<LawsPage />} />
                                <Route path="/about" element={<AboutPage />} />
                            </Routes>
                        </Box>
                    </Box>
                </BrowserRouter>
            </ThemeProvider>
        </ColorModeContext.Provider>
    );
}

export default App;
