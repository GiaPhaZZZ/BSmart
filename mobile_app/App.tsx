/**
 * BSmart App Entry Point
 */

import { StyleSheet, View } from 'react-native';
import { SafeAreaProvider } from 'react-native-safe-area-context';
import { MainScreen } from './src/screens/MainScreen';

function App() {
  return (
    <SafeAreaProvider style={styles.container}>
      <View style={styles.container}>
        <MainScreen />
      </View>
    </SafeAreaProvider>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#010409',
  },
});

export default App;
