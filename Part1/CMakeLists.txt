#CMake entry point
cmake_minimum_required (VERSION 2.6)
project (Tutorials)

set(OpenGL_GL_PREFERENCE
		GLVND
		)
find_package(OpenGL REQUIRED)


set(ALL_LIBS
  ${OPENGL_LIBRARY}
	GLEW
	glfw
	pthread
)

add_definitions(
	-DTW_STATIC
	-DTW_NO_LIB_PRAGMA
	-DTW_NO_DIRECT3D
	-DGLEW_STATIC
	-D_CRT_SECURE_NO_WARNINGS
)


add_executable(code01_dessin
	Cours1/code01_dessin.cpp Common/shaders_utilities.cpp
)
target_link_libraries(code01_dessin
	${ALL_LIBS}
)
add_custom_command(
  TARGET code01_dessin POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/code01_dessin${CMAKE_EXECUTABLE_SUFFIX}" "${CMAKE_CURRENT_SOURCE_DIR}/Cours1/"
  )

add_executable(code02_dessin
	Cours1/code02_dessin.cpp Common/shaders_utilities.cpp
)
target_link_libraries(code02_dessin
	${ALL_LIBS}
)

add_custom_command(
		TARGET code02_dessin POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/code02_dessin${CMAKE_EXECUTABLE_SUFFIX}" "${CMAKE_CURRENT_SOURCE_DIR}/Cours1/"
)

add_executable(code03_dessin
	Cours1/code03_dessin.cpp Common/shaders_utilities.cpp
)
target_link_libraries(code03_dessin
	${ALL_LIBS}
)

add_custom_command(
		TARGET code03_dessin POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/code03_dessin${CMAKE_EXECUTABLE_SUFFIX}" "${CMAKE_CURRENT_SOURCE_DIR}/Cours1/"
)

add_executable(code04_dessin
	Cours1/code04_dessin.cpp Common/shaders_utilities.cpp
)
target_link_libraries(code04_dessin
	${ALL_LIBS}
)

add_custom_command(
		TARGET code04_dessin POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/code04_dessin${CMAKE_EXECUTABLE_SUFFIX}" "${CMAKE_CURRENT_SOURCE_DIR}/Cours1/"
)
add_executable(code05_dessin
	Cours1/code05_dessin.cpp Common/shaders_utilities.cpp
)
target_link_libraries(code05_dessin
	${ALL_LIBS}
)
add_custom_command(
  TARGET code05_dessin POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/code05_dessin${CMAKE_EXECUTABLE_SUFFIX}" "${CMAKE_CURRENT_SOURCE_DIR}/Cours1/"
  )

add_executable(code05_exo
	Cours1/code05_exo.cpp Common/shaders_utilities.cpp
)

target_link_libraries(code05_exo
	${ALL_LIBS}
)

add_custom_command(
	TARGET code05_exo POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/code05_exo${CMAKE_EXECUTABLE_SUFFIX}" "${CMAKE_CURRENT_SOURCE_DIR}/Cours1/"
)


add_executable(lapin
	Cours1/lapin.cpp Common/shaders_utilities.cpp Cours1/modele.cpp
)
target_link_libraries(lapin
	${ALL_LIBS}
)
add_custom_command(
  TARGET lapin POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/lapin${CMAKE_EXECUTABLE_SUFFIX}" "${CMAKE_CURRENT_SOURCE_DIR}/Cours1/"
  )


add_executable(code05_ico
	Cours1/code05_ico.cpp Common/shaders_utilities.cpp Cours1/code05_ico.cpp
)
target_link_libraries(code05_ico
	${ALL_LIBS}
)
add_custom_command(
  TARGET code05_ico POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/code05_ico${CMAKE_EXECUTABLE_SUFFIX}" "${CMAKE_CURRENT_SOURCE_DIR}/Cours1/"
  )
  
add_executable(code01_trajectoire
    TD_Icosaedre/LecteurTraj/code01_trajectoire.cpp TD_Icosaedre/LecteurTraj/lecture.cpp Common/shaders_utilities.cpp Common/lecture_trajectoire.cpp Common/xdrfile.c
)
target_link_libraries(code01_trajectoire
    ${ALL_LIBS}
)

add_custom_command(
  TARGET code01_trajectoire POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_CFG_INTDIR}/code01_trajectoire${CMAKE_EXECUTABLE_SUFFIX}" "${CMAKE_CURRENT_SOURCE_DIR}/TD_Icosaedre/LecteurTraj"
  )
