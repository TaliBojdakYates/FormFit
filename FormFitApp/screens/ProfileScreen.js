import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, ScrollView, Switch, Image } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import { COLORS, FONTS, SHADOW, BORDER_RADIUS } from '../globalStyles';

const ProfileScreen = () => {
    // Sample user data
    const [userData, setUserData] = useState({
        name: 'Alex Johnson',
        email: 'alex.johnson@example.com',
        profilePicture: null,
        preferences: {
            darkMode: false,
            notifications: true,
            dataSharing: false,
            units: 'imperial',
        },
        stats: {
            totalWorkouts: 28,
            favoriteExercise: 'Squat',
            highestScore: 94,
        },
        weight: 185, // Weight in lbs
        height: '5\'10"', // Height in feet and inches
        subscription: {
            type: 'Pro',
            renewalDate: '2025-04-15',
        }
    });

    // Toggle switch handlers
    const toggleDarkMode = () => {
        setUserData(prev => ({
            ...prev,
            preferences: {
                ...prev.preferences,
                darkMode: !prev.preferences.darkMode
            }
        }));
    };

    const toggleNotifications = () => {
        setUserData(prev => ({
            ...prev,
            preferences: {
                ...prev.preferences,
                notifications: !prev.preferences.notifications
            }
        }));
    };

    const toggleDataSharing = () => {
        setUserData(prev => ({
            ...prev,
            preferences: {
                ...prev.preferences,
                dataSharing: !prev.preferences.dataSharing
            }
        }));
    };

    const toggleUnits = () => {
        setUserData(prev => ({
            ...prev,
            preferences: {
                ...prev.preferences,
                units: prev.preferences.units === 'imperial' ? 'metric' : 'imperial'
            }
        }));
    };

    // Render a settings item with a switch
    const renderSwitchItem = (icon, title, value, onToggle) => (
        <View style={styles.settingItem}>
            <View style={styles.settingItemLeft}>
                <View style={styles.settingIcon}>
                    <Ionicons name={icon} size={24} color="#4F46E5" />
                </View>
                <Text style={styles.settingText}>{title}</Text>
            </View>
            <Switch
                value={value}
                onValueChange={onToggle}
                trackColor={{ false: '#D1D5DB', true: '#4F46E5' }}
                thumbColor="white"
            />
        </View>
    );

    // Render a settings item with a chevron
    const renderChevronItem = (icon, title, subtitle, onPress) => (
        <TouchableOpacity style={styles.settingItem} onPress={onPress}>
            <View style={styles.settingItemLeft}>
                <View style={styles.settingIcon}>
                    <Ionicons name={icon} size={24} color="#4F46E5" />
                </View>
                <View>
                    <Text style={styles.settingText}>{title}</Text>
                    {subtitle && <Text style={styles.settingSubtext}>{subtitle}</Text>}
                </View>
            </View>
            <Ionicons name="chevron-forward" size={20} color="#9CA3AF" />
        </TouchableOpacity>
    );

    return (
        <SafeAreaView style={styles.container}>
            <ScrollView contentContainerStyle={styles.scrollContent}>
                {/* Profile Header */}
                <View style={styles.profileHeader}>
                    <View style={styles.profilePhotoContainer}>
                        <View style={styles.profilePhotoPlaceholder}>
                            <Text style={styles.profileInitials}>
                                {userData.name.split(' ').slice(0, 2).map(n => n[0]).join('')}
                            </Text>
                        </View>
                    </View>
                    <Text style={styles.profileName}>{userData.name}</Text>
                    <Text style={styles.profileEmail}>{userData.email}</Text>
                    <TouchableOpacity style={styles.editProfileButton}>
                        <Text style={styles.editProfileText}>Edit Profile</Text>
                    </TouchableOpacity>
                </View>

                {/* Stats Section */}
                <View style={styles.statsContainer}>
                    <View style={styles.statsRow}>
                        <View style={styles.statBox}>
                            <Text style={styles.statValue}>{userData.stats.totalWorkouts}</Text>
                            <Text style={styles.statLabel}>Workouts</Text>
                        </View>
                        <View style={styles.statBox}>
                            <Text style={styles.statValue}>{userData.stats.highestScore}</Text>
                            <Text style={styles.statLabel}>Best Score</Text>
                        </View>
                    </View>
                </View>

                {/* Personal Info Section */}
                <View style={styles.personalInfoContainer}>
                    <View style={styles.personalInfoItem}>
                        <Text style={styles.personalInfoLabel}>Weight</Text>
                        <Text style={styles.personalInfoValue}>{userData.weight} lbs</Text>
                    </View>
                    <View style={styles.personalInfoItem}>
                        <Text style={styles.personalInfoLabel}>Height</Text>
                        <Text style={styles.personalInfoValue}>{userData.height}</Text>
                    </View>
                </View>


                {/* Settings Section */}
                <View style={styles.settingsContainer}>

                    <TouchableOpacity style={styles.signOutButton}>
                        <Ionicons name="log-out-outline" size={20} color={COLORS.textPrimary} />
                        <Text style={styles.signOutText}>Sign Out</Text>
                    </TouchableOpacity>
                </View>
            </ScrollView>
        </SafeAreaView>
    );
};

export default ProfileScreen;

const styles = StyleSheet.create({
    container: {
        flex: 1,
        backgroundColor: COLORS.background,

    },
    scrollContent: {
        paddingBottom: 90,
        padding: 20,
    },
    profileHeader: {
        alignItems: 'center',
        marginBottom: 30,
        marginTop: 20,
    },
    profilePhotoContainer: {
        width: 120,
        height: 120,
        borderRadius: 60,
        overflow: 'hidden',
        backgroundColor: COLORS.primary,
        position: 'relative',
        borderWidth: 4,
        borderColor: COLORS.textPrimary,
    },
    profilePhoto: {
        width: '100%',
        height: '100%',
        borderRadius: 60,
    },
    profilePhotoPlaceholder: {
        width: '100%',
        height: '100%',
        backgroundColor: COLORS.textSecondary,
        justifyContent: 'center',
        alignItems: 'center',
        borderRadius: 60,
    },
    profileInitials: {
        fontSize: 40,
        color: COLORS.textPrimary,
        fontWeight: 'bold',
    },
    editPhotoButton: {
        position: 'absolute',
        bottom: 0,
        right: 0,
        backgroundColor: COLORS.primary,
        borderRadius: 20,
        padding: 8,
    },
    profileName: {
        fontSize: FONTS.heading,
        fontWeight: '700',
        color: COLORS.textPrimary,
        marginTop: 10,
    },
    profileEmail: {
        fontSize: FONTS.small,
        color: COLORS.borderColor,
        marginTop: 5,
    },
    editProfileButton: {
        marginTop: 10,
        backgroundColor: COLORS.primary,
        paddingVertical: 10,
        paddingHorizontal: 25,
        borderRadius: BORDER_RADIUS,
    },
    editProfileText: {
        fontSize: FONTS.body,
        color: COLORS.textPrimary,
        fontWeight: '600',
    },

    statsContainer: {
        marginTop: 20,
        marginBottom: 30,
    },
    statsRow: {
        flexDirection: 'row',
        justifyContent: 'space-between',
    },
    statBox: {
        flex: 1,
        backgroundColor: COLORS.cardBackground,
        paddingVertical: 15,
        paddingHorizontal: 20,
        borderRadius: BORDER_RADIUS,
        marginHorizontal: 10,
        justifyContent: 'center',
        alignItems: 'center',
        ...SHADOW,
    },
    statValue: {
        fontSize: 32,
        fontWeight: '600',
        color: COLORS.primary,
    },
    statLabel: {
        fontSize: FONTS.small,
        color: COLORS.textSecondary,
    },

    personalInfoContainer: {
        marginTop: 20,
        marginBottom: 30,
    },
    personalInfoItem: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        paddingVertical: 12,
        paddingHorizontal: 20,
        borderRadius: BORDER_RADIUS,
        backgroundColor: COLORS.cardBackground,
        marginBottom: 15,
        ...SHADOW,
    },
    personalInfoLabel: {
        fontSize: FONTS.body,
        color: COLORS.textPrimary,
    },
    personalInfoValue: {
        fontSize: FONTS.body,
        color: COLORS.textSecondary,
        fontWeight: '600',
    },

    subscriptionContainer: {
        marginBottom: 30,
        backgroundColor: COLORS.cardBackground,
        padding: 20,
        borderRadius: BORDER_RADIUS,
        ...SHADOW,
    },
    subscriptionHeader: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        marginBottom: 20,
    },
    subscriptionTitle: {
        fontSize: FONTS.body,
        fontWeight: '600',
        color: COLORS.textPrimary,
    },
    subscriptionRenewal: {
        fontSize: FONTS.small,
        color: COLORS.textSecondary,
    },
    subscriptionBadge: {
        backgroundColor: COLORS.success,
        borderRadius: 12,
        paddingVertical: 4,
        paddingHorizontal: 10,
    },
    subscriptionBadgeText: {
        color: 'white',
        fontWeight: '700',
    },
    subscriptionButton: {
        backgroundColor: COLORS.primary,
        paddingVertical: 10,
        paddingHorizontal: 25,
        borderRadius: BORDER_RADIUS,
        alignItems: 'center',
    },
    subscriptionButtonText: {
        fontSize: FONTS.body,
        color: 'white',
        fontWeight: '600',
    },

    settingsContainer: {
        marginTop: 20,
        marginBottom: 30,
    },
    signOutButton: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        backgroundColor: '#EF4444',
        paddingVertical: 12,
        paddingHorizontal: 20,
        borderRadius: BORDER_RADIUS,
        ...SHADOW,
    },
    signOutText: {
        fontSize: FONTS.body,
        color: COLORS.textPrimary,
        marginLeft: 10,
    },
});
