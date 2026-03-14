import React from 'react';
import { Box } from '@mui/material';
import { motion } from 'framer-motion';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome'; // Magician/Sparkle icon looks more "AI Magic" than a robot toy

const AnimatedAvatar = ({ isThinking = false, size = 60, iconSize = 35 }) => {

    // Floating Logic
    const floatVariant = {
        animate: {
            y: [0, -8, 0],
            rotate: [0, 5, -5, 0],
            transition: {
                duration: 6,
                repeat: Infinity,
                ease: "easeInOut"
            }
        }
    };

    // Pulse / Glow Logic
    // We will overlay a glowing gradient

    return (
        <Box
            component={motion.div}
            variants={floatVariant}
            animate="animate"
            sx={{
                position: 'relative',
                width: size,
                height: size,
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center'
            }}
        >
            {/* Outer Glow Ring */}
            <Box
                component={motion.div}
                animate={{
                    scale: [1, 1.2, 1],
                    opacity: [0.5, 0.2, 0.5],
                }}
                transition={{
                    duration: 3,
                    repeat: Infinity,
                    ease: "easeInOut"
                }}
                sx={{
                    position: 'absolute',
                    width: '100%',
                    height: '100%',
                    borderRadius: '50%',
                    background: 'radial-gradient(circle, rgba(255,193,7,0.4) 0%, rgba(255,193,7,0) 70%)', // Amber cleaning
                    zIndex: 0,
                }}
            />

            {/* Main Avatar Circle */}
            <Box
                sx={{
                    width: '100%',
                    height: '100%',
                    borderRadius: '50%',
                    background: 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)', // Dark Slate Gradient
                    border: '2px solid #ffc107', // Amber Border
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                    zIndex: 1,
                    boxShadow: '0 4px 20px rgba(0,0,0,0.3)',
                }}
            >
                {/* Inner Icon with Shine */}
                <motion.div
                    animate={{
                        rotate: isThinking ? 360 : 0,
                        scale: isThinking ? 0.9 : 1
                    }}
                    transition={{
                        rotate: { duration: 2, repeat: Infinity, ease: "linear" },
                        scale: { duration: 0.5, repeat: Infinity, repeatType: "reverse" }
                    }}
                >
                    <AutoAwesomeIcon sx={{ fontSize: iconSize, color: '#ffc107' }} /> {/* Amber Icon */}
                </motion.div>
            </Box>
        </Box>
    );
};

export default AnimatedAvatar;
