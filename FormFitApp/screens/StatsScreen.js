import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, ScrollView, Dimensions, SafeAreaView } from 'react-native';
import { useRoute } from '@react-navigation/native';
import { LineChart } from 'react-native-chart-kit';
import { COLORS, FONTS } from '../globalStyles';

const StatsScreen = () => {
    const route = useRoute();
    const [analysisData, setAnalysisData] = useState(null);

    useEffect(() => {
        if (route.params?.data) {
            setAnalysisData(route.params.data);

        }
    }, [route.params]);

    const chartData = {
        labels: analysisData?.labels || [],
        datasets: [
            {
                data: analysisData?.leftKneeAngles || [],
                strokeWidth: 2,
                color: () => '#FF6384',
                withDots: false,
                withShadow: false,
            },
            {
                data: analysisData?.rightKneeAngles || [],
                strokeWidth: 2,
                color: () => '#36A2EB', // Solid color for right knee (RGB: 54, 162, 235)
                withDots: false,
                withShadow: false,
            },
        ],
    };

    // Function to determine score color based on value
    const getScoreColor = (score) => {
        if (score >= 90) return '#10B981'; // Green for excellent
        if (score >= 75) return '#F59E0B'; // Yellow/Orange for good
        return '#EF4444'; // Red for needs improvement
    };

    const screenWidth = Dimensions.get('window').width;
    const chartWidth = screenWidth - 64; // Subtract 16 for padding

    return (
        <SafeAreaView style={styles.container}>
            <ScrollView style={styles.scrollContent}>

                {analysisData ? (
                    <View style={styles.analysisContainer}>
                        <View style={styles.analysisHeader}>
                            <Text style={styles.exerciseTitle}>Analysis</Text>
                            <Text style={styles.analysisDate}>{analysisData.date}</Text>
                        </View>
                        {/* Score Card Section */}
                        <View style={styles.scoreCard}>
                            <Text style={styles.scoreTitle}>Score</Text>
                            <View style={styles.scoreGaugeContainer}>
                                <View
                                    style={[styles.scoreGauge, { backgroundColor: getScoreColor(analysisData.score) }]}
                                >
                                    <Text style={styles.scoreValue}>{analysisData.score}</Text>
                                </View>
                            </View>
                            <View style={styles.scoreLabels}>
                                <Text style={styles.scoreRangeLabel}>Low</Text>
                                <Text style={styles.scoreRangeLabel}>High</Text>
                            </View>
                        </View>

                        {/* Chart Section */}
                        <View style={styles.chartContainer}>
                            <Text style={styles.chartTitle}>Performance Over Time</Text>
                            <LineChart
                                data={chartData}
                                width={chartWidth}
                                height={220}
                                chartConfig={{
                                    backgroundColor: COLORS.background,
                                    backgroundGradientFrom: COLORS.cardBackground,
                                    backgroundGradientTo: COLORS.cardBackground,
                                    decimalPlaces: 2,
                                    color: (opacity = 1) => `rgba(255, 255, 255, ${opacity})`,
                                    labelColor: (opacity = 1) => `rgba(255, 255, 255, ${opacity})`,
                                    style: {
                                        borderRadius: 16,
                                    },
                                    propsForLabels: {
                                        fontSize: 10,
                                    },
                                }}
                                style={styles.chart}
                                withHorizontalLines={false}
                                withVerticalLines={false}
                                withInnerLines={false}
                                withShadow={false}
                            />
                        </View>
                    </View>
                ) : (
                    <View style={styles.emptyState}>
                        <Text style={styles.emptyStateText}>No Current Analysis</Text>
                        <Text style={styles.emptyStateSubtext}>Please check back later.</Text>
                    </View>
                )}
                <View style={{ padding: 45 }}></View>
            </ScrollView>
        </SafeAreaView>
    );
};

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: COLORS.background,

    },
    scrollContent: {
        paddingBottom: 90,


    },
    analysisContainer: {
        padding: 16,
        gap: 16,

    },
    analysisHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
    },
    exerciseTitle: {
        fontSize: FONTS.heading,
        fontWeight: 'bold',
        color: COLORS.textPrimary,
    },
    analysisDate: {
        fontSize: FONTS.small,
        color: COLORS.textSecondary,
    },
    chartContainer: {
        backgroundColor: COLORS.cardBackground,
        borderRadius: 8,
        padding: 16,
        elevation: 5,
        shadowColor: COLORS.shadowColor,
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.25,
        shadowRadius: 3.5,

    },
    chartTitle: {
        fontSize: FONTS.subheading,
        fontWeight: 'bold',
        color: COLORS.textPrimary,
        marginBottom: 8,
        textAlign: 'left',  // Align text to the left
    },
    chart: {
        borderRadius: 16,
    },
    scoreCard: {
        backgroundColor: COLORS.cardBackground,
        borderRadius: 8,
        padding: 16,
        elevation: 5,
        shadowColor: COLORS.shadowColor,
        shadowOffset: { width: 0, height: 4 },
        shadowOpacity: 0.25,
        shadowRadius: 3.5,
    },
    scoreTitle: {
        fontSize: FONTS.subheading,
        fontWeight: 'bold',
        color: COLORS.textPrimary,
        marginBottom: 16,
    },
    scoreGaugeContainer: {
        alignItems: 'center',
    },
    scoreGauge: {
        width: 120,
        height: 120,
        borderRadius: 60,
        borderWidth: 6, // Outer white line
        borderColor: '#FFFFFF',
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: 16,
    },

    scoreValue: {
        fontSize: 36,
        fontWeight: 'bold',
        color: '#FFFFFF',
    },
    scoreLabels: {
        flexDirection: 'row',
        width: '100%',
        justifyContent: 'space-between',
    },
    scoreRangeLabel: {
        fontSize: FONTS.small,
        color: COLORS.textSecondary,
    },
    emptyState: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        padding: 32,
    },
    emptyStateText: {
        fontSize: FONTS.heading,
        fontWeight: 'bold',
        color: COLORS.textPrimary,
        marginTop: 16,
    },
    emptyStateSubtext: {
        fontSize: FONTS.body,
        color: COLORS.textSecondary,
        textAlign: 'center',
        marginTop: 8,
    },
});

export default StatsScreen;
