import { createTheme } from '@mui/material/styles';

const getDesignTokens = (mode) => ({
  palette: {
    mode,
    primary: {
      main: '#1e293b', // Deep Slate Blue
      light: '#475569',
      dark: '#0f172a',
    },
    secondary: {
      main: '#ffc107', // Amber
      light: '#ffca28',
      dark: '#ff8f00',
      contrastText: '#000000',
    },
    background: {
      default: mode === 'light' ? '#f8fafc' : '#0f172a', // Light Slate vs Dark Slate
      paper: mode === 'light' ? '#ffffff' : '#1e293b',
    },
    text: {
      primary: mode === 'light' ? '#0f172a' : '#ffffff', // Darker Slate / Pure White
      secondary: mode === 'light' ? '#334155' : '#e2e8f0', // Darker Slate / Bright Slate
    },
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif', // Added Inter
    fontSize: 15, // Base size up from 14
    h4: {
      fontWeight: 700,
      letterSpacing: '-0.02em',
    },
    h5: {
      fontWeight: 600,
      letterSpacing: '-0.01em',
    },
    h6: {
      fontWeight: 600,
    },
    body1: {
      fontSize: '1.05rem', // ~17px
      lineHeight: 1.6,
    },
    body2: {
      fontSize: '0.95rem', // ~15px
      lineHeight: 1.5,
    },
    button: {
      fontWeight: 600,
    },
  },
  shape: {
    borderRadius: 8,
  },
  components: {
    MuiButton: {
      styleOverrides: {
        root: {
          textTransform: 'none',
          fontWeight: 500,
        },
      },
    },
    MuiPaper: {
      styleOverrides: {
        root: {
          backgroundImage: 'none', // Fix for dark mode elevation opacity
          boxShadow: mode === 'light' ? '0 2px 8px rgba(0,0,0,0.1)' : '0 2px 8px rgba(0,0,0,0.3)',
        },
      },
    },
  },
});

export default getDesignTokens;
