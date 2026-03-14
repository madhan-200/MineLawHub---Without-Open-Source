import React from 'react';
import { Box, Typography, Paper, Grid, Button, Chip } from '@mui/material';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import GavelIcon from '@mui/icons-material/Gavel';
import ChatIcon from '@mui/icons-material/Chat';

const laws = [
    {
        name: 'The Mines Act, 1952',
        shortName: 'MinesAct1952',
        year: 1952,
        color: '#1565c0',
        description: 'Principal legislation regulating labour and safety in mines. Covers employment, health, safety, inspections, and penalties.',
        keySections: ['Section 17 – Manager duties', 'Section 40 – Employment age', 'Section 46 – Women underground', 'Section 74 – Penalties'],
        query: 'What is the Mines Act 1952?',
    },
    {
        name: 'MCDR 2017',
        shortName: 'MCDR_2017',
        year: 2017,
        color: '#2e7d32',
        description: 'Mineral Conservation and Development Rules. Governs systematic mining, mineral conservation, and submission of mining plans.',
        keySections: ['Rule 23 – Mining plan', 'Rule 27 – Annual returns', 'Rule 34 – Penalties', 'Rule 45 – Inspections'],
        query: 'What are MCDR 2017 rules?',
    },
    {
        name: 'Coal Mines Regulations, 2017',
        shortName: 'CoalMinesReg',
        year: 2017,
        color: '#4e342e',
        description: 'Comprehensive regulations for coal mine safety including ventilation, support, winding operations, and emergency procedures.',
        keySections: ['Regulation 17 – Safety committee', 'Regulation 99 – Ventilation', 'Regulation 182 – Winding', 'Regulation 186 – Safety lamps'],
        query: 'What are Coal Mines Regulations 2017?',
    },
    {
        name: 'MMDR Act, 1957',
        shortName: 'mmdr_act,1957',
        year: 1957,
        color: '#6a1b9a',
        description: 'Mines and Minerals (Development and Regulation) Act. Framework for grant of mining leases, royalties, and mineral development.',
        keySections: ['Section 4 – Prospecting licence', 'Section 5 – Mining lease', 'Section 9 – Royalties', 'Section 21 – Penalties'],
        query: 'What is MMDR Act 1957?',
    },
    {
        name: 'The Mines Rules, 1955',
        shortName: 'MinesRules1955',
        year: 1955,
        color: '#e65100',
        description: 'Detailed rules under the Mines Act covering working hours, leave, welfare amenities, and mine management structure.',
        keySections: ['Rule 29 – Working hours', 'Rule 30 – Overtime', 'Rule 40 – Welfare', 'Rule 59 – Reports'],
        query: 'What are the Mines Rules 1955?',
    },
    {
        name: 'DGMS Technical Circulars',
        shortName: 'dgms_tech_circular',
        year: 2024,
        color: '#c62828',
        description: 'Technical circulars issued by the Director General of Mines Safety containing safety guidelines and operational directives.',
        keySections: ['Safety procedures', 'Operational guidelines', 'Technical standards', 'Compliance requirements'],
        query: 'What are DGMS safety circulars?',
    },
];

const LawsPage = () => {
    const navigate = useNavigate();

    const handleAsk = (query) => {
        navigate(`/chat?q=${encodeURIComponent(query)}`);
    };

    return (
        <Box sx={{ flex: 1, overflow: 'auto', bgcolor: 'background.default' }}>
            {/* Header */}
            <Box sx={{ textAlign: 'center', py: 5, px: 3 }}>
                <motion.div initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}>
                    <GavelIcon sx={{ fontSize: 50, color: 'secondary.main', mb: 1 }} />
                    <Typography variant="h4" sx={{ fontWeight: 800, color: 'text.primary', mb: 1 }}>
                        Laws & Regulations Reference
                    </Typography>
                    <Typography variant="body1" sx={{ color: 'text.secondary', maxWidth: 600, mx: 'auto' }}>
                        MineLawHub covers 6 Indian mining legislations. Explore each Act below and click "Ask About This Act" to start a conversation.
                    </Typography>
                </motion.div>
            </Box>

            {/* Law Cards */}
            <Box sx={{ maxWidth: 1100, mx: 'auto', px: 3, pb: 6 }}>
                <Grid container spacing={3}>
                    {laws.map((law, i) => (
                        <Grid item xs={12} sm={6} md={4} key={law.shortName}>
                            <motion.div
                                initial={{ opacity: 0, y: 30 }}
                                animate={{ opacity: 1, y: 0 }}
                                transition={{ delay: i * 0.1 }}
                            >
                                <Paper
                                    elevation={0}
                                    sx={{
                                        p: 3,
                                        height: '100%',
                                        display: 'flex',
                                        flexDirection: 'column',
                                        border: '1px solid',
                                        borderColor: 'divider',
                                        borderRadius: 3,
                                        borderTop: `4px solid ${law.color}`,
                                        transition: 'all 0.3s ease',
                                        '&:hover': {
                                            transform: 'translateY(-4px)',
                                            boxShadow: '0 8px 30px rgba(0,0,0,0.12)',
                                        },
                                    }}
                                >
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1 }}>
                                        <Typography variant="h6" sx={{ fontWeight: 700, color: 'text.primary', fontSize: '1rem' }}>
                                            {law.name}
                                        </Typography>
                                        <Chip label={law.year} size="small" sx={{ bgcolor: law.color + '18', color: law.color, fontWeight: 600 }} />
                                    </Box>

                                    <Typography variant="body2" sx={{ color: 'text.secondary', mb: 2, lineHeight: 1.5, flexGrow: 1 }}>
                                        {law.description}
                                    </Typography>

                                    <Box sx={{ mb: 2 }}>
                                        <Typography variant="caption" sx={{ fontWeight: 600, color: 'text.secondary', display: 'block', mb: 0.5 }}>
                                            Key Sections:
                                        </Typography>
                                        {law.keySections.map((s) => (
                                            <Typography key={s} variant="caption" sx={{ display: 'block', color: 'text.secondary', pl: 1 }}>
                                                • {s}
                                            </Typography>
                                        ))}
                                    </Box>

                                    <Button
                                        variant="outlined"
                                        size="small"
                                        startIcon={<ChatIcon />}
                                        onClick={() => handleAsk(law.query)}
                                        sx={{
                                            borderColor: law.color,
                                            color: law.color,
                                            fontWeight: 600,
                                            borderRadius: 2,
                                            '&:hover': { bgcolor: law.color + '10', borderColor: law.color },
                                        }}
                                    >
                                        Ask About This Act →
                                    </Button>
                                </Paper>
                            </motion.div>
                        </Grid>
                    ))}
                </Grid>
            </Box>
        </Box>
    );
};

export default LawsPage;
