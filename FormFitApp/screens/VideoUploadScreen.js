import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Image, ActivityIndicator, ScrollView, FlatList } from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';
import { Ionicons } from '@expo/vector-icons';
import * as ImagePicker from 'expo-image-picker';
import { Video } from 'expo-av';
import { useNavigation } from '@react-navigation/native';

import { COLORS, FONTS, SHADOW, BORDER_RADIUS } from '../globalStyles';

const VideoUploadScreen = () => {
    const [video, setVideo] = useState(null);
    const [analyzing, setAnalyzing] = useState(false);
    const [analyzed, setAnalyzed] = useState(false);
    const [exercise, setExercise] = useState('Squat'); // State for the selected exercise type
    const navigation = useNavigation();


    const [isOpen, setIsOpen] = useState(false);
    const [selectedItem, setSelectedItem] = useState('Squat');

    const data = [
        { label: 'Squat', value: 'Squat' },
        { label: 'Bench Press', value: 'Bench Press' },
        { label: 'Deadlift', value: 'Deadlift' },
    ];

    const toggleDropdown = () => {
        setIsOpen(!isOpen);
    };

    const selectItem = (item) => {
        setSelectedItem(item);
        setIsOpen(false);
        setExercise(item);
    };

    // Request permissions and select video
    const pickVideo = async () => {
        const { status } = await ImagePicker.requestMediaLibraryPermissionsAsync();
        if (status !== 'granted') {
            alert('Sorry, we need camera roll permissions to make this work!');
            return;
        }

        let result = await ImagePicker.launchImageLibraryAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.Videos,
            allowsEditing: true,
            quality: 1,
        });

        if (!result.canceled) {
            setVideo(result.assets[0].uri);
            setAnalyzed(false); // Reset the analyzed state
        }
    };

    // Record video using the camera
    const recordVideo = async () => {
        const { status } = await ImagePicker.requestCameraPermissionsAsync();
        if (status !== 'granted') {
            alert('Sorry, we need camera permissions to make this work!');
            return;
        }

        let result = await ImagePicker.launchCameraAsync({
            mediaTypes: ImagePicker.MediaTypeOptions.Videos,
            allowsEditing: true,
            quality: 1,
            videoMaxDuration: 60,
        });

        if (!result.canceled) {
            setVideo(result.assets[0].uri);
            setAnalyzed(false); // Reset the analyzed state
        }
    };

    // Function to upload video to Flask API
    const uploadVideo = async () => {
        if (!video) return;

        setAnalyzing(true);

        // Get file name and type information
        const uriParts = video.split('.');
        const fileType = uriParts[uriParts.length - 1];

        const formData = new FormData();
        formData.append('video', {
            uri: video,
            name: `video.${fileType}`,
            type: `video/${fileType}`,

        });
        formData.append('exercise', exercise);
        try {
            const response = await fetch('http://104.194.116.241:5000/upload-video', {
                method: 'POST',
                body: formData,
                headers: {
                    'Content-Type': 'multipart/form-data',
                },
            });

            const result = await response.json();
            if (response.ok) {
                setAnalyzing(false);
                setAnalyzed(true); // Set to true once analyzed
                // Navigate with the selected exercise and video data
                navigation.navigate('Stats', {
                    source: 'video',
                    data: {
                        date: new Date().toISOString(),
                        exercise: exercise, // Send the selected exercise type
                        leftKneeAngles: result.left_knee_angles,
                        rightKneeAngles: result.right_knee_angles,
                        score: result.score
                    }
                });
            } else {
                setAnalyzing(false);
                alert('Error analyzing video: ' + result.message);
            }
        } catch (error) {
            setAnalyzing(false);
            alert('Error uploading video: ' + error.message);
            console.log('Error:', error);
        }
    };

    return (
        <SafeAreaView style={styles.container}>
            <ScrollView contentContainerStyle={styles.scrollContent}>
                <View style={styles.header}>
                    <Text style={styles.title}>Upload Workout Video</Text>
                    <Text style={styles.subtitle}>
                        Upload or record a video of your workout to analyze your form
                    </Text>
                </View>



                <View style={styles.videoContainer}>
                    {video ? (
                        <View style={styles.videoWrapper}>
                            <Video
                                source={{ uri: video }}
                                style={styles.video}
                                useNativeControls
                                resizeMode="contain"
                            />
                            <TouchableOpacity
                                style={styles.changeButton}
                                onPress={pickVideo}
                            >
                                <Text style={styles.changeButtonText}>Choose Different Video</Text>
                            </TouchableOpacity>
                        </View>
                    ) : (
                        <View style={styles.uploadPlaceholder}>
                            <Ionicons name="cloud-upload-outline" size={48} color="#9CA3AF" />
                            <Text style={styles.uploadText}>No video selected</Text>
                        </View>
                    )}
                </View>

                <View style={styles.buttonContainer}>
                    {!video && (
                        <>
                            <TouchableOpacity
                                style={[styles.button, styles.uploadButton]}
                                onPress={pickVideo}
                            >
                                <Ionicons name="cloud-upload-outline" size={24} color="white" />
                                <Text style={styles.buttonText}>Select from Library</Text>
                            </TouchableOpacity>

                            <TouchableOpacity
                                style={[styles.button, styles.recordButton]}
                                onPress={recordVideo}
                            >
                                <Ionicons name="videocam-outline" size={24} color="white" />
                                <Text style={styles.buttonText}>Record Video</Text>
                            </TouchableOpacity>
                        </>
                    )}

                    {video && !analyzing && !analyzed && (
                        <>
                            <View style={styles.container}>
                                <TouchableOpacity onPress={toggleDropdown} style={styles.dropdownButton}>
                                    <Text style={styles.selectedText}>{selectedItem}</Text>
                                </TouchableOpacity>

                                {isOpen && (
                                    <View style={styles.dropdownList}>
                                        <ScrollView contentContainerStyle={styles.dropdownList}>
                                            {data?.map((item, index) => (
                                                <TouchableOpacity key={item.value || index} onPress={() => selectItem(item.label)} style={styles.dropdownItem}>
                                                    <Text style={styles.itemText}>{item.label}</Text>
                                                </TouchableOpacity>
                                            ))}
                                        </ScrollView>
                                    </View>

                                )}
                            </View>

                            <TouchableOpacity
                                style={[styles.button, styles.analyzeButton]}
                                onPress={uploadVideo}
                            >
                                <Ionicons name="analytics-outline" size={24} color="white" />
                                <Text style={styles.buttonText}>Analyze Form</Text>
                            </TouchableOpacity>
                        </>
                    )}

                    {analyzing && (
                        <View style={styles.analyzingContainer}>
                            <ActivityIndicator size="large" color="#4F46E5" />
                            <Text style={styles.analyzingText}>Analyzing your form...</Text>
                        </View>
                    )}
                </View>

                <View style={styles.infoSection}>
                    <View style={styles.infoCard}>
                        <Ionicons name="bulb-outline" size={24} color="#4F46E5" />
                        <View style={styles.infoTextContainer}>
                            <Text style={styles.infoTitle}>Tips for Better Analysis</Text>
                            <View style={styles.tipsList}>
                                <View style={styles.tipItem}>
                                    <View style={styles.tipBullet} />
                                    <Text style={styles.tipText}>Record in front for best results</Text>
                                </View>
                                <View style={styles.tipItem}>
                                    <View style={styles.tipBullet} />
                                    <Text style={styles.tipText}>Ensure your full body is visible</Text>
                                </View>
                                <View style={styles.tipItem}>
                                    <View style={styles.tipBullet} />
                                    <Text style={styles.tipText}>Use good lighting</Text>
                                </View>
                                <View style={styles.tipItem}>
                                    <View style={styles.tipBullet} />
                                    <Text style={styles.tipText}>Record at least 3 reps of the exercise</Text>
                                </View>
                            </View>
                        </View>
                    </View>
                </View>
            </ScrollView>
        </SafeAreaView>
    );
};

export default VideoUploadScreen;



const styles = StyleSheet.create({

    container: {
        flex: 1,
        backgroundColor: COLORS.background,

    },
    scrollContent: {
        paddingBottom: 90,

    },
    header: {
        padding: 16,
        backgroundColor: COLORS.secondary,
    },
    title: {
        fontSize: FONTS.heading,
        fontWeight: 'bold',
        color: COLORS.textPrimary,
        marginBottom: 8,
    },
    subtitle: {
        fontSize: FONTS.body,
        color: COLORS.textSecondary,
    },
    videoContainer: {
        padding: 16,
    },
    videoWrapper: {
        backgroundColor: COLORS.cardBackground,
        borderRadius: BORDER_RADIUS,
        overflow: 'hidden',
    },
    video: {
        height: 240,
        borderRadius: BORDER_RADIUS,
    },
    changeButton: {
        backgroundColor: COLORS.textSecondary,
        padding: 8,
        alignItems: 'center',
    },
    dropdownButton: {
        backgroundColor: '#fff',
        borderRadius: 5,
        padding: 10,
        borderWidth: 1,
        borderColor: '#ccc',
    },
    selectedText: {
        fontSize: 16,
        color: '#333',
    },
    dropdownList: {
        backgroundColor: '#fff',
        borderRadius: 5,
        borderWidth: 1,
        borderColor: '#ccc',
        marginTop: 5,
        maxHeight: 200, // Max height to avoid overflow
        overflow: 'scroll',
    },
    dropdownItem: {
        padding: 12,
    },
    itemText: {
        fontSize: 16,
        color: '#333',
    },

    changeButtonText: {
        color: COLORS.textPrimary,
        fontWeight: 'bold',
    },
    uploadPlaceholder: {
        backgroundColor: COLORS.cardBackground,
        height: 240,
        borderRadius: BORDER_RADIUS,
        justifyContent: 'center',
        alignItems: 'center',
    },
    uploadText: {
        marginTop: 16,
        color: COLORS.textSecondary,
        fontSize: FONTS.body,
    },
    buttonContainer: {
        padding: 16,
        gap: 12,
    },
    button: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        padding: 16,
        borderRadius: BORDER_RADIUS,
        gap: 8,
    },
    uploadButton: {
        backgroundColor: COLORS.primary,
    },
    recordButton: {
        backgroundColor: COLORS.success,
    },
    analyzeButton: {
        backgroundColor: '#F59E0B', // Amber color for analyze action (not in COLORS, you may want to add it)
    },
    buttonText: {
        color: COLORS.textPrimary,
        fontWeight: 'bold',
        fontSize: FONTS.body,
    },
    analyzingContainer: {
        alignItems: 'center',
        justifyContent: 'center',
        padding: 16,
    },
    analyzingText: {
        marginTop: 16,
        fontSize: FONTS.body,
        color: COLORS.textSecondary,
    },
    infoSection: {
        padding: 16,
        flex: 1,
        justifyContent: 'flex-end',
    },
    infoCard: {
        backgroundColor: COLORS.secondary,
        borderRadius: BORDER_RADIUS,
        padding: 16,
        flexDirection: 'row',
        alignItems: 'flex-start',
        ...SHADOW,
    },
    infoTextContainer: {
        marginLeft: 12,
        flex: 1,
    },
    infoTitle: {
        fontSize: FONTS.body,
        fontWeight: 'bold',
        color: COLORS.textPrimary,
        marginBottom: 8,
    },
    tipsList: {
        gap: 8,
    },
    tipItem: {
        flexDirection: 'row',
        alignItems: 'center',
    },
    tipBullet: {
        width: 6,
        height: 6,
        borderRadius: 9999,
        backgroundColor: COLORS.primary,
        marginRight: 8,
    },
    tipText: {
        fontSize: FONTS.small,
        color: COLORS.textSecondary,
    },
});