/**
 * BSmart App Entry Point
 */

import { StyleSheet, View } from 'react-native';
import { MainScreen } from './src/screens/MainScreen';

function App() {
  return (
    <View style={styles.container}>
      <MainScreen />
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#010409',
  },
});

export default App;
