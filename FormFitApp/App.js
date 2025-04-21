import React from 'react';
import { View, StyleSheet } from 'react-native';
import { NavigationContainer } from '@react-navigation/native';
import { createBottomTabNavigator } from '@react-navigation/bottom-tabs';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { StatusBar } from 'react-native';
import { Ionicons } from '@expo/vector-icons';

import VideoUploadScreen from './screens/VideoUploadScreen';
import StatsScreen from './screens/StatsScreen';
import ProfileScreen from './screens/ProfileScreen';
import { COLORS, FONTS, BORDER_RADIUS, SHADOW } from './globalStyles';

const Tab = createBottomTabNavigator();

export default function App() {
  return (
    <SafeAreaProvider>
      <StatusBar barStyle="light-content" backgroundColor={COLORS.background} />
      <NavigationContainer>
        <Tab.Navigator
          screenOptions={({ route }) => ({
            tabBarIcon: ({ focused, color, size }) => {
              let iconName;

              if (route.name === 'Upload') {
                iconName = focused ? 'cloud-upload' : 'cloud-upload-outline';
              } else if (route.name === 'Stats') {
                iconName = focused ? 'stats-chart' : 'stats-chart-outline';
              } else if (route.name === 'Profile') {
                iconName = focused ? 'person' : 'person-outline';
              }

              return (
                <Ionicons
                  name={iconName}
                  size={focused ? size + 2 : size}
                  color={focused ? COLORS.success : COLORS.textSecondary}
                />
              );
            },
            tabBarActiveTintColor: COLORS.primary,
            tabBarInactiveTintColor: COLORS.textSecondary,
            tabBarShowLabel: false,
            tabBarStyle: styles.tabBar,
            headerShown: true,
            headerStyle: styles.header,
            headerTitleStyle: styles.headerTitle,
            headerTintColor: COLORS.lightText,
            headerShadowVisible: false,
          })}
        >
          <Tab.Screen name="Upload" component={VideoUploadScreen} />
          <Tab.Screen name="Stats" component={StatsScreen} />

        </Tab.Navigator>
      </NavigationContainer>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  tabBar: {
    backgroundColor: '#2D3B46',

    borderRadius: BORDER_RADIUS * 2,
    height: 70,
    position: 'absolute',
    left: 16,
    right: 16,
    paddingTop: 15,
    bottom: 16,
    elevation: 8,
    borderTopWidth: 0,
    paddingBottom: 0,
    alignItems: 'center',
    justifyContent: 'center',
    marginHorizontal: 16,
    ...SHADOW,
  },
  header: {
    backgroundColor: COLORS.background,
    borderBottomWidth: 0,
    elevation: 0,
    height: 60,
  },
  headerTitle: {
    fontWeight: '600',
    fontSize: FONTS.subheading,
    color: COLORS.lightText,
  },
});