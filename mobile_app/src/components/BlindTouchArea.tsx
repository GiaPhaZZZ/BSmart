/**
 * BSmart Blind Touch Area Component
 * Large-surface interactive area designed specifically for visually impaired users.
 * Supports:
 *   - Hold to speak (Push-to-talk)
 *   - Release to send
 *   - Quick tap or 2-finger touch to cancel / return to IDLE
 * Fully accessible with Android TalkBack.
 */

import React, { useRef } from 'react';
import {
  AccessibilityInfo,
  GestureResponderEvent,
  PanResponder,
  PanResponderGestureState,
  StyleSheet,
  Text,
  Vibration,
  View,
} from 'react-native';
import { AppState } from '../types';

interface BlindTouchAreaProps {
  appState: AppState;
  onHold: () => void;
  onRelease: () => void;
  onShortPress: () => void;
}

export function BlindTouchArea({
  appState,
  onHold,
  onRelease,
  onShortPress,
}: BlindTouchAreaProps) {
  const touchStartTime = useRef<number>(0);
  const isHolding = useRef<boolean>(false);
  const holdTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const panResponder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => true,
      onMoveShouldSetPanResponder: () => true,

      onPanResponderGrant: (evt: GestureResponderEvent) => {
        touchStartTime.current = Date.now();
        isHolding.current = false;

        // Check for 2-finger touch -> Emergency Cancel / IDLE
        if (evt.nativeEvent.touches && evt.nativeEvent.touches.length >= 2) {
          AccessibilityInfo.announceForAccessibility('Chạm hai ngón tay: Hủy khẩn cấp');
          onShortPress();
          return;
        }

        // Trigger Hold after 250ms of holding with tactile haptic feedback
        holdTimer.current = setTimeout(() => {
          isHolding.current = true;
          try {
            Vibration.vibrate(60);
          } catch {}
          onHold();
        }, 250);
      },

      onPanResponderRelease: (
        _evt: GestureResponderEvent,
        _gestureState: PanResponderGestureState,
      ) => {
        if (holdTimer.current) {
          clearTimeout(holdTimer.current);
          holdTimer.current = null;
        }

        const duration = Date.now() - touchStartTime.current;

        if (isHolding.current) {
          // User held the screen and is now releasing
          isHolding.current = false;
          onRelease();
        } else if (duration < 250) {
          // Quick tap -> cancel / return to IDLE
          onShortPress();
        }
      },

      onPanResponderTerminate: () => {
        if (holdTimer.current) {
          clearTimeout(holdTimer.current);
          holdTimer.current = null;
        }
        if (isHolding.current) {
          isHolding.current = false;
          onRelease();
        }
      },
    }),
  ).current;

  const isListening = appState === AppState.LISTENING;
  const isProcessing = appState === AppState.PROCESSING;

  const getBackgroundColor = () => {
    if (isListening) return '#B71C1C'; // Bright Red when listening
    if (isProcessing) return '#E65100'; // Amber when processing
    return '#102A43'; // Deep Navy Blue for ready/idle
  };

  const getBorderColor = () => {
    if (isListening) return '#FF5252';
    if (isProcessing) return '#FFB74D';
    return '#4FC3F7';
  };

  const getPrimaryInstruction = () => {
    if (isListening) return 'Đang lắng nghe...\n(Thả tay ra để gửi)';
    if (isProcessing) return 'Đang xử lý AI...\n(Vui lòng chờ)';
    return 'CHẠM & GIỮ ĐỂ NÓI\n(Thả tay để gửi lệnh)';
  };

  return (
    <View
      {...panResponder.panHandlers}
      accessible={true}
      accessibilityRole="button"
      accessibilityLabel={`Màn hình tương tác người khiếm thị. Trạng thái hiện tại: ${appState}. Chạm và giữ bất kỳ đâu để nói lệnh. Thả tay ra để gửi. Chạm nhanh một lần hoặc hai ngón tay để hủy và về trang chủ.`}
      accessibilityActions={[
        { name: 'activate', label: 'Bắt đầu nói lệnh' },
        { name: 'escape', label: 'Hủy hoặc về trang chủ' },
      ]}
      onAccessibilityAction={event => {
        if (event.nativeEvent.actionName === 'activate') {
          try {
            Vibration.vibrate(60);
          } catch {}
          onHold();
        } else if (event.nativeEvent.actionName === 'escape') {
          onShortPress();
        }
      }}
      style={[
        styles.container,
        {
          backgroundColor: getBackgroundColor(),
          borderColor: getBorderColor(),
        },
      ]}
    >
      <View style={styles.touchAreaInner}>
        {/* Visual Wave / Pulse Indicator */}
        <View
          style={[
            styles.micIndicator,
            {
              backgroundColor: isListening
                ? '#FF1744'
                : isProcessing
                  ? '#FFA000'
                  : '#0288D1',
            },
          ]}
        >
          <Text style={styles.micIconText}>
            {isListening ? '🎙️' : isProcessing ? '⚡' : '👆'}
          </Text>
        </View>

        <Text style={styles.primaryText}>{getPrimaryInstruction()}</Text>

        <View style={styles.helpBox}>
          <Text style={styles.helpText}>
            • Chạm giữ: Nói lệnh (Tính năng 1, 2, 3)
          </Text>
          <Text style={styles.helpText}>
            • Thả tay: Gửi lệnh đến kính
          </Text>
          <Text style={styles.helpText}>
            • Chạm 2 lần: Hủy lệnh khẩn cấp
          </Text>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    minHeight: 280,
    borderRadius: 20,
    borderWidth: 3,
    padding: 20,
    justifyContent: 'center',
    alignItems: 'center',
    marginVertical: 12,
    elevation: 6,
    shadowColor: '#4FC3F7',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
  },
  touchAreaInner: {
    alignItems: 'center',
    justifyContent: 'center',
    width: '100%',
  },
  micIndicator: {
    width: 80,
    height: 80,
    borderRadius: 40,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 16,
    borderWidth: 2,
    borderColor: '#FFFFFF',
  },
  micIconText: {
    fontSize: 36,
  },
  primaryText: {
    color: '#FFFFFF',
    fontSize: 22,
    fontWeight: '800',
    textAlign: 'center',
    lineHeight: 30,
    letterSpacing: 0.5,
    textTransform: 'uppercase',
  },
  helpBox: {
    marginTop: 20,
    paddingVertical: 10,
    paddingHorizontal: 14,
    backgroundColor: 'rgba(0, 0, 0, 0.4)',
    borderRadius: 12,
    width: '100%',
  },
  helpText: {
    color: '#E0E0E0',
    fontSize: 13,
    fontWeight: '500',
    marginVertical: 2,
  },
});
