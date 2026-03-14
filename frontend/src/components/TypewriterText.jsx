import React from 'react';
import { motion } from 'framer-motion';
import { Typography } from '@mui/material';

const TypewriterText = ({ text, variant = "body1", speed = 0.03, ...props }) => {
    const container = {
        hidden: { opacity: 0 },
        visible: (i = 1) => ({
            opacity: 1,
            transition: { staggerChildren: speed, delayChildren: 0.04 * i },
        }),
    };

    const child = {
        visible: {
            opacity: 1,
            y: 0,
            transition: {
                type: "spring",
                damping: 12,
                stiffness: 100,
            },
        },
        hidden: {
            opacity: 0,
            y: 5,
            transition: {
                type: "spring",
                damping: 12,
                stiffness: 100,
            },
        },
    };

    return (
        <Typography
            component={motion.div}
            variants={container}
            initial="hidden"
            animate="visible"
            variant={variant}
            {...props}
            style={{ display: 'inline-block' }} // important for width calculation
        >
            {text.split("").map((letter, index) => (
                <motion.span variants={child} key={index}>
                    {letter}
                </motion.span>
            ))}
        </Typography>
    );
};

export default TypewriterText;
