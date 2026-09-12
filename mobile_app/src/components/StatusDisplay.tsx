/**
 * BSmart Status Display Component
 * Shows current app state with dark high-tech theme.
 * See: docs/requirement.md §8
 */

import React, { useEffect, useRef } from 'react';
import { Animated, StyleSheet, Text, View } from 'react-native';
import { AppState } from '../types';

const STATE_LABELS: Record<AppState, string> = {
  [AppState.IDLE]: 'Chờ lệnh',
  [AppState.LISTENING]: 'Đang nghe...',
  [AppState.PROCESSING]: 'Đang xử lý...',
  [AppState.FEATURE_1_QA]: 'Tính năng 1 — Hỏi đáp',
  [AppState.FEATURE_2_CAPTURE]: 'Tính năng 2 — Chụp ảnh',
  [AppState.FEATURE_3_NAVIGATION]: 'Tính năng 3 — Cảnh báo vật cản',
};

const STATE_COLORS: Record<AppState, string> = {
  [AppState.IDLE]: '#4FC3F7',
  [AppState.LISTENING]: '#FF7043',
  [AppState.PROCESSING]: '#FFB300',
  [AppState.FEATURE_1_QA]: '#AB47BC',
  [AppState.FEATURE_2_CAPTURE]: '#66BB6A',
  [AppState.FEATURE_3_NAVIGATION]: '#26C6DA',
};

interface StatusDisplayProps {
  appState: AppState;
}

export function StatusDisplay({ appState }: StatusDisplayProps) {
  const pulseAnim = useRef(new Animated.Value(1)).current;

  useEffect(() => {
    if (
      appState === AppState.LISTENING ||
      appState === AppState.PROCESSING ||
      appState === AppState.FEATURE_3_NAVIGATION
    ) {
      const pulse = Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, {
            toValue: 1.15,
            duration: 600,
            useNativeDriver: true,
          }),
          Animated.timing(pulseAnim, {
            toValue: 1,
            duration: 600,
            useNativeDriver: true,
          }),
        ]),
      );
      pulse.start();
      return () => pulse.stop();
    } else {
      pulseAnim.setValue(1);
    }
  }, [appState, pulseAnim]);

  const color = STATE_COLORS[appState];

  return (
    <View style={styles.container}>
      <Animated.View
        style={[
          styles.indicator,
          { backgroundColor: color, transform: [{ scale: pulseAnim }] },
        ]}
      />
      <View style={styles.labelContainer}>
        <Text style={styles.stateLabel}>{STATE_LABELS[appState]}</Text>
        <Text style={[styles.stateCode, { color }]}>{appState}</Text>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 20,
    paddingVertical: 16,
    backgroundColor: '#0D1117',
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#21262D',
    marginBottom: 12,
  },
  indicator: {
    width: 14,
    height: 14,
    borderRadius: 7,
    marginRight: 14,
  },
  labelContainer: {
    flex: 1,
  },
  stateLabel: {
    color: '#E6EDF3',
    fontSize: 18,
    fontWeight: '600',
    letterSpacing: 0.4,
  },
  stateCode: {
    fontSize: 11,
    fontWeight: '400',
    marginTop: 2,
    letterSpacing: 1,
    textTransform: 'uppercase',
    opacity: 0.7,
  },
});
